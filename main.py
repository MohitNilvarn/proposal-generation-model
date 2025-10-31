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

proposal_prompt_template = """You are a professional proposal writer for AppSynergies Pvt Ltd, a software development company.

CRITICAL INSTRUCTIONS:
1. Generate a PROFESSIONAL, STRUCTURED PROPOSAL in PROSE format - NOT bullet points
2. Follow the structure of professional consulting proposals with clear sections
3. Use ONLY information from the provided context - DO NOT invent or hallucinate details
4. If pricing information is not in the context, DO NOT make up numbers - state "Pricing to be determined based on detailed requirements"
5. Write in complete paragraphs and sentences, NOT lists or bullet points
6. Include proper section headers using markdown (##)
7. Be specific and reference actual details from the context when available
8. Maintain a professional, confident tone throughout

CONTEXT FROM KNOWLEDGE BASE:
{context}

CLIENT REQUIREMENTS:
{requirements}

PROPOSAL STRUCTURE TO FOLLOW:
## Executive Summary
Brief overview of the project and proposed solution (2-3 paragraphs)

## Project Understanding
Detailed understanding of client requirements and objectives (2-4 paragraphs)

## Proposed Solution
Comprehensive description of the solution approach, including:
- Technical architecture and approach
- Key features and functionalities
- Technology stack recommendations
- Integration requirements

## Project Deliverables
Clear description of what will be delivered (in prose, not bullet points)

## Development Approach
Explanation of methodology, team structure, and workflow

## Timeline and Phases
Project timeline with phase descriptions (can use a brief phase breakdown table if needed)

## Team Composition
Description of the team and their roles

## Investment and Pricing
If pricing information is available in context, present it clearly. If not, state that detailed pricing will be provided after requirements analysis.

## Support and Maintenance
Post-deployment support details

## Why AppSynergies
Company strengths and differentiators

## Next Steps
Clear actions for moving forward

IMPORTANT REMINDERS:
- Write in PROSE with complete sentences and paragraphs
- NO bullet point lists in the main content
- If information is missing from context, acknowledge it professionally
- Use only factual information from the provided context
- Maintain professional consulting proposal style throughout

Generate the proposal now:"""

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

    # Improved chunking for better context preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Increased chunk size
        chunk_overlap=400,  # Increased overlap
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
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)  # Slightly increased temp for better prose

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
    """Format requirements in a more narrative style"""
    parts = []
    
    if query.question:
        parts.append(f"**Client Request:** {query.question}\n")
    
    # Group related intent data
    technical_aspects = []
    business_aspects = []
    
    for key, val in intent.items():
        if not val:
            continue
            
        formatted_key = key.replace('_', ' ').title()
        formatted_val = ', '.join(val) if isinstance(val, list) else val
        
        if key in ['platforms', 'tech_stack', 'platform_type', 'automations']:
            technical_aspects.append(f"{formatted_key}: {formatted_val}")
        else:
            business_aspects.append(f"{formatted_key}: {formatted_val}")
    
    if business_aspects:
        parts.append("**Business Requirements:**\n" + "\n".join(f"- {aspect}" for aspect in business_aspects))
    
    if technical_aspects:
        parts.append("\n**Technical Requirements:**\n" + "\n".join(f"- {aspect}" for aspect in technical_aspects))
    
    # Add structured query fields
    additional = []
    for k, v in query.dict().items():
        if k == "question" or not v:
            continue
        formatted_val = ', '.join(v) if isinstance(v, list) else v
        additional.append(f"{k.replace('_', ' ').title()}: {formatted_val}")
    
    if additional:
        parts.append("\n**Additional Details:**\n" + "\n".join(f"- {item}" for item in additional))
    
    return "\n".join(parts)

def validate_and_clean_proposal(text: str, context: str) -> tuple[str, str]:
    """
    Validate proposal and assign confidence score
    Returns: (cleaned_text, confidence_level)
    """
    # Check for hallucination indicators
    hallucination_indicators = [
        r'\$[\d,]+',  # Dollar amounts not in context
        r'Rs\.?\s*[\d,]+',  # Rupee amounts not in context
        r'\d+\s*(?:months?|weeks?|days?)',  # Timeframes not in context
    ]
    
    confidence = "HIGH"
    
    # Extract numbers from proposal
    proposal_numbers = set(re.findall(r'\d+(?:,\d+)*', text))
    context_numbers = set(re.findall(r'\d+(?:,\d+)*', context))
    
    # Check if proposal mentions numbers not in context
    fabricated_numbers = proposal_numbers - context_numbers
    if len(fabricated_numbers) > 3:  # Allow some tolerance
        confidence = "MEDIUM"
    
    # Check for placeholder phrases
    placeholders = [
        "to be determined",
        "will be provided",
        "contact for pricing",
        "based on requirements"
    ]
    
    if any(phrase in text.lower() for phrase in placeholders):
        confidence = "MEDIUM"
    
    # Remove any remaining placeholder brackets
    text = re.sub(r'\[Information[^\]]*\]', '', text)
    text = re.sub(r'\[TBD\]', 'To be determined', text, flags=re.IGNORECASE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    return text, confidence

# ========== Routes ==========
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "✅ Proposal Generator ready",
        "endpoints": ["/ask"],
        "version": "2.0-improved"
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
        
        # Retrieve more context with better search
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 12,  # Increased from 8
                "fetch_k": 20  # Fetch more candidates for MMR
            }
        )
        related_docs = retriever.invoke(combined_text)

        if not related_docs:
            return {
                "answer": "Unfortunately, I don't have sufficient information in the knowledge base to generate a proposal for this request. Please provide more details or contact AppSynergies directly at info@AppSynergies.com",
                "extracted_intent": intent,
                "confidence_score": "LOW"
            }

        # Build context with document sources
        context = "\n\n---\n\n".join(d.page_content for d in related_docs)
        
        # Generate proposal
        prompt = proposal_prompt.format(context=context, requirements=requirements)
        resp = llm.invoke(prompt)
        
        # Validate and clean
        cleaned_answer, confidence = validate_and_clean_proposal(
            resp.content or "",
            context
        )
        
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