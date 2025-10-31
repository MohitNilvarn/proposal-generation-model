import os
import re
import json
import shutil
import base64
from dotenv import load_dotenv
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from google.oauth2 import service_account
from googleapiclient.discovery import build

# ========== Load Environment ==========
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_DOC_ID = os.getenv("GOOGLE_DOC_ID")

CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in environment")
if not GOOGLE_DOC_ID:
    raise ValueError("❌ Missing GOOGLE_DOC_ID in environment")
if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise ValueError("❌ Missing GOOGLE_SERVICE_ACCOUNT_JSON in environment")

# ========== Decode Google Credentials ==========
try:
    try:
        creds_json = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    except json.JSONDecodeError:
        decoded = base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds_json = json.loads(decoded)

    GOOGLE_CREDS = service_account.Credentials.from_service_account_info(
        creds_json,
        scopes=["https://www.googleapis.com/auth/documents.readonly"]
    )
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Google credentials: {e}")

# ========== FastAPI App ==========
app = FastAPI(title="Proposal Generator - Google Doc KB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Models ==========
class Query(BaseModel):
    question: Optional[str] = None
    industry: Optional[str] = None
    ecommerce_type: Optional[str] = None
    platforms: Optional[List[str]] = None
    tech_focus: Optional[str] = None
    service_type: Optional[str] = None
    budget_range: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    extracted_intent: dict | None = None
    confidence_score: Optional[str] = None

# ========== Enhanced Prompts ==========
intent_extraction_prompt = """You are an expert at extracting structured information from user requests.

USER INPUT:
{question}

Extract and return ONLY a valid JSON object with these keys (use null for missing values):
- platforms: list of technology platforms mentioned
- vendor_type: type of vendor/business model
- project_types: types of projects or features needed
- automations: automation requirements
- platform_type: web, mobile, or both
- tech_stack: specific technologies requested
- services: types of services needed
- industry: business industry/domain
- ecommerce_type: type of ecommerce (if applicable)
- budget_range: budget mentioned (if any)

Return ONLY the JSON, no other text."""

# IMPROVED PROPOSAL PROMPT - More explicit instructions to prevent hallucination
proposal_prompt_template = """You are a professional proposal writer for AppSynergies Pvt Ltd, a software development company.

⚠️ CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY use information explicitly stated in the CONTEXT below
2. If pricing/timeline information is NOT in the context, write: "Pricing will be provided after detailed requirement analysis"
3. If technical details are NOT in the context, write: "Technical specifications will be finalized during the requirement gathering phase"
4. DO NOT invent features, prices, timelines, or technical specifications
5. When in doubt, acknowledge the gap rather than making up information

📋 FORMATTING REQUIREMENTS:
1. Write in PROFESSIONAL PROSE with complete sentences and paragraphs
2. NO bullet points in main sections (only use for deliverables if absolutely necessary)
3. Each section should have 2-4 well-developed paragraphs
4. Make each point slightly elaborative - add ONE sentence of context/explanation per key point
5. Use proper section headers with markdown (##)

CONTEXT FROM KNOWLEDGE BASE:
{context}

CLIENT REQUIREMENTS:
{requirements}

PROPOSAL STRUCTURE:

## Executive Summary
Write 2-3 paragraphs providing an overview of the project opportunity. Begin by acknowledging the client's needs as expressed in their requirements. Then briefly describe AppSynergies' understanding of the project scope and the proposed approach. Conclude with a statement about the expected outcomes and value proposition. Make this engaging and client-focused.

## Project Understanding
Write 2-4 paragraphs demonstrating deep understanding of the client's requirements. Start by restating the core business problem or opportunity. Then explain how the proposed solution aligns with their objectives, elaborating on specific requirements they've mentioned. Discuss any industry-specific considerations or challenges that AppSynergies recognizes. This should show that you truly understand their needs.

## Proposed Solution
Write 3-5 paragraphs describing the comprehensive solution. Begin with the overall technical approach and architecture philosophy. Then elaborate on the key features and functionalities, explaining how each addresses specific client needs (add one sentence of context for each major feature). Discuss the technology stack recommendations with brief justifications. Finally, cover integration requirements and how the solution will work within their existing ecosystem.

## Project Deliverables
Write 2-3 paragraphs clearly describing what will be delivered. Instead of listing items, describe the deliverables in narrative form, grouping related items together. For each major deliverable, add one sentence explaining its purpose or value. Explain the completeness and quality standards that will be applied.

## Development Approach
Write 2-3 paragraphs explaining the methodology and workflow. Describe the development methodology (Agile, iterative, etc.) and why it's appropriate. Elaborate on team structure and how collaboration will work, including client involvement at key stages. Explain quality assurance processes and how feedback will be incorporated.

## Timeline and Phases
Write 2-3 paragraphs describing the project timeline. Explain the phased approach and why it's structured this way. For each major phase, provide a brief description of what will be accomplished and approximately how long it will take (ONLY if this information is in the context). If timeline details are not in the context, state: "A detailed timeline will be provided after requirement finalization."

## Team Composition
Write 2 paragraphs describing the team. Explain the roles that will be involved in the project and their responsibilities. Elaborate on the team's expertise and how their skills align with project needs. Add context about collaboration and communication approaches.

## Investment and Pricing
Write 2-3 paragraphs about pricing. If specific pricing is available in the context, present it clearly with explanations of what each component covers. If pricing is NOT in the context, write: "Detailed pricing will be provided after a thorough requirement analysis session. Our pricing is structured to be transparent and aligned with the value delivered. We will provide a comprehensive breakdown covering design, development, testing, deployment, and ongoing support."

## Support and Maintenance
Write 2 paragraphs about post-deployment support. Describe the support model and what it covers, elaborating on response times and coverage. Explain maintenance services and how they ensure long-term success.

## Why AppSynergies
Write 2-3 paragraphs highlighting company strengths. Explain what makes AppSynergies the right partner for this project, elaborating on relevant experience, expertise, and differentiators. Focus on aspects that are particularly relevant to the client's needs.

## Next Steps
Write 1-2 paragraphs outlining clear next steps. Describe the immediate actions needed to move forward, such as requirement gathering sessions, kickoff meetings, and contract finalization. Make this actionable and time-bound where appropriate.

---

IMPORTANT VALIDATION BEFORE WRITING:
- Review the context carefully
- Identify what information IS available
- Identify what information is NOT available
- For missing information, use the placeholder phrases provided above
- Ensure every factual statement can be traced back to the context

Now generate the proposal following all instructions above:"""

intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=proposal_prompt_template, input_variables=["context", "requirements"])

# ========== Google Docs Helpers ==========
def fetch_google_doc_text(document_id: str, creds) -> str:
    service = build("docs", "v1", credentials=creds, cache_discovery=False)
    doc = service.documents().get(documentId=document_id).execute()
    content = []

    def read_elements(elements):
        for el in elements:
            if "paragraph" in el:
                parts = el["paragraph"].get("elements", [])
                para_text = "".join(p.get("textRun", {}).get("content", "") for p in parts).strip()
                if para_text:
                    content.append(para_text)
            elif "table" in el:
                for row in el["table"].get("tableRows", []):
                    cells = []
                    for cell in row.get("tableCells", []):
                        cell_text = " ".join(
                            c.get("paragraph", {}).get("elements", [])[0]
                            .get("textRun", {})
                            .get("content", "")
                            for c in cell.get("content", [])
                            if "paragraph" in c
                        ).strip()
                        cells.append(cell_text)
                    content.append(" | ".join(cells))
            elif "sectionBreak" in el:
                content.append("\n")

    read_elements(doc.get("body", {}).get("content", []))
    return "\n\n".join([c for c in content if c.strip()])

# ========== Embeddings & VectorStore ==========
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def build_or_load_vectorstore_from_google_doc():
    if os.path.exists(CHROMA_DIR):
        try:
            return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception:
            shutil.rmtree(CHROMA_DIR)

    print("📄 Fetching Google Doc content...")
    text = fetch_google_doc_text(GOOGLE_DOC_ID, GOOGLE_CREDS)
    if not text.strip():
        raise RuntimeError("Google Doc is empty")

    # Improved chunking strategy for better context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,  # Larger chunks for better context
        chunk_overlap=500,  # More overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    print(f"📚 Indexing {len(docs)} chunks...")
    vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    try:
        vs.persist()
    except Exception:
        pass
    return vs

vectorstore = build_or_load_vectorstore_from_google_doc()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)  # Lower temp for less hallucination

# ========== Utils ==========
def extract_intent_from_text(text: str) -> dict:
    try:
        resp = llm.invoke(intent_prompt.format(question=text))
        content = resp.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception as e:
        print(f"⚠️ Intent extraction error: {e}")
        return {}

def format_requirements(intent: dict, query: Query) -> str:
    """Format requirements in a clear, structured way"""
    parts = []
    
    if query.question:
        parts.append(f"**Primary Request:** {query.question}\n")
    
    # Technical requirements
    technical_items = []
    for key in ['platforms', 'tech_stack', 'platform_type', 'automations', 'project_types']:
        val = intent.get(key)
        if val:
            formatted_key = key.replace('_', ' ').title()
            formatted_val = ', '.join(val) if isinstance(val, list) else val
            technical_items.append(f"• {formatted_key}: {formatted_val}")
    
    if technical_items:
        parts.append("**Technical Requirements:**\n" + "\n".join(technical_items))
    
    # Business requirements
    business_items = []
    for key in ['vendor_type', 'services', 'industry', 'ecommerce_type', 'budget_range']:
        val = intent.get(key)
        if val:
            formatted_key = key.replace('_', ' ').title()
            formatted_val = ', '.join(val) if isinstance(val, list) else val
            business_items.append(f"• {formatted_key}: {formatted_val}")
    
    # Add from query object
    for k, v in query.dict().items():
        if k != "question" and v:
            formatted_key = k.replace('_', ' ').title()
            formatted_val = ', '.join(v) if isinstance(v, list) else v
            business_items.append(f"• {formatted_key}: {formatted_val}")
    
    if business_items:
        parts.append("\n**Business Requirements:**\n" + "\n".join(business_items))
    
    return "\n".join(parts)

def validate_proposal_against_context(proposal: str, context: str) -> tuple[str, str, list]:
    """
    Enhanced validation to detect hallucination
    Returns: (cleaned_proposal, confidence_score, warning_list)
    """
    warnings = []
    confidence = "HIGH"
    
    # Extract all numbers from both texts
    proposal_numbers = set(re.findall(r'\$\s*[\d,]+|\₹\s*[\d,]+|USD\s*[\d,]+|INR\s*[\d,]+|\d+\s*(?:USD|INR)', proposal))
    context_numbers = set(re.findall(r'\$\s*[\d,]+|\₹\s*[\d,]+|USD\s*[\d,]+|INR\s*[\d,]+|\d+\s*(?:USD|INR)', context))
    
    # Check for numbers not in context
    fabricated_numbers = proposal_numbers - context_numbers
    if fabricated_numbers:
        warnings.append(f"⚠️ Found pricing not in knowledge base: {', '.join(list(fabricated_numbers)[:3])}")
        confidence = "MEDIUM"
    
    # Check for vague timeline promises not in context
    timeline_patterns = [
        r'\d+\s*(?:weeks?|months?|days?)\s*(?:timeline|duration|period)',
        r'(?:within|in)\s*\d+\s*(?:weeks?|months?|days?)'
    ]
    
    for pattern in timeline_patterns:
        proposal_timelines = re.findall(pattern, proposal.lower())
        context_timelines = re.findall(pattern, context.lower())
        if proposal_timelines and not context_timelines:
            warnings.append("⚠️ Timeline estimates not found in knowledge base")
            confidence = "MEDIUM"
            break
    
    # Check for generic tech stack mentions not in context
    tech_terms = ['flutter', 'react', 'next.js', 'firebase', 'mongodb', 'mysql', 'aws', 'node.js']
    proposal_lower = proposal.lower()
    context_lower = context.lower()
    
    mentioned_tech = [tech for tech in tech_terms if tech in proposal_lower and tech not in context_lower]
    if len(mentioned_tech) > 2:
        warnings.append(f"⚠️ Technologies mentioned not in knowledge base: {', '.join(mentioned_tech)}")
        confidence = "LOW"
    
    # Clean up any placeholder artifacts
    cleaned = re.sub(r'\[.*?\]', '', proposal)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    
    return cleaned, confidence, warnings

# ========== Routes ==========
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "✅ Enhanced Proposal Generator ready",
        "endpoints": ["/ask", "/refresh-kb"],
        "version": "3.0-anti-hallucination"
    }

@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    combined_text = "\n".join([
        query.question or "",
        *[f"{k}: {v}" for k, v in query.dict().items() if v and k != "question"]
    ])
    
    if not combined_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide a question or structured input"
        )

    try:
        # Extract intent
        intent = extract_intent_from_text(combined_text)
        requirements = format_requirements(intent, query)
        
        # Retrieve MORE context with better search strategy
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 15,  # Get more chunks
                "fetch_k": 30,  # Consider more candidates
                "lambda_mult": 0.7  # Balance relevance vs diversity
            }
        )
        related_docs = retriever.invoke(combined_text)

        if not related_docs:
            return {
                "answer": "Unfortunately, I don't have sufficient information in the knowledge base to generate a detailed proposal for this request. Please provide more specific details about your project requirements, or contact AppSynergies directly at info@appsynergies.com for a customized proposal.",
                "extracted_intent": intent,
                "confidence_score": "LOW"
            }

        # Build comprehensive context
        context = "\n\n---DOCUMENT SECTION---\n\n".join(d.page_content for d in related_docs)
        
        # Add explicit instruction about what's in context
        context_summary = f"""
AVAILABLE INFORMATION SUMMARY:
- Number of relevant sections found: {len(related_docs)}
- Context covers: {', '.join(set([doc.page_content.split()[0] for doc in related_docs[:5]]))}

FULL CONTEXT:
{context}
"""
        
        # Generate proposal with enhanced context
        prompt = proposal_prompt.format(context=context_summary, requirements=requirements)
        resp = llm.invoke(prompt)
        
        # Validate against context
        cleaned_answer, confidence, warnings = validate_proposal_against_context(
            resp.content or "",
            context
        )
        
        # Add warnings to response if any
        if warnings:
            warning_section = "\n\n---\n**Validation Notes:**\n" + "\n".join(warnings)
            cleaned_answer += warning_section
        
        return {
            "answer": cleaned_answer,
            "extracted_intent": intent,
            "confidence_score": confidence
        }
        
    except Exception as e:
        print(f"❌ Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/refresh-kb")
def refresh_knowledge_base():
    """Endpoint to manually refresh the knowledge base"""
    try:
        global vectorstore
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        vectorstore = build_or_load_vectorstore_from_google_doc()
        return {"status": "success", "message": "Knowledge base refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh KB: {str(e)}")