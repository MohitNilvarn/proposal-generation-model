from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load Word document
DOCX_PATH = "Proposal Knowledge Base.docx"
loader = Docx2txtLoader(DOCX_PATH)
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", "!", "?", ","]
)
splits = text_splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
vector_db = Chroma.from_documents(splits, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 12})

# Enhanced prompt template
template = """You are a professional proposal generation assistant for a software development company. Your role is to create detailed, accurate proposals based EXCLUSIVELY on the provided knowledge base.

**CRITICAL ANTI-HALLUCINATION RULES:**
1. ONLY use information explicitly stated in the retrieved context below
2. If information is not in the context, state: "This feature is not covered in our standard offerings"
3. Never invent features, pricing, or technical details
4. Match project types EXACTLY to templates in the knowledge base
5. Preserve exact pricing amounts and currency formats (USD, INR, Rs)
6. Do not add features from similar projects unless explicitly requested

**USER REQUIREMENTS:**
Platforms: {platforms}
Vendor Type: {vendor_type}
Project Type: {project_type}
Automations: {automations}
AI Platforms: {ai_platforms}
Platform Category: {platform_category}
Technology: {tech_type}
Services: {services}
Custom Requirements: {custom_prompt}

**RETRIEVED KNOWLEDGE BASE CONTEXT:**
{context}

**OUTPUT REQUIREMENTS:**
Generate a comprehensive proposal with these sections:

1. **PROJECT OVERVIEW**
   - Brief introduction based on user's selected platforms and project type
   - Scope alignment with knowledge base

2. **ADMIN PANEL FEATURES**
   - List ALL relevant admin features from the matched project template
   - Elaborate each feature with 1-2 sentences explaining functionality
   - Group related features logically

3. **APPLICATION/WEBSITE FEATURES**
   - List ALL relevant app/website features from the template
   - Elaborate each feature with implementation details
   - Highlight unique capabilities

4. **TECHNOLOGY STACK**
   - Specify technologies exactly as listed in knowledge base
   - Include frontend, backend, hosting details

5. **PRICING BREAKDOWN**
   - Design costs (exact amount and currency)
   - Development costs (exact amount and currency)
   - AI/ML integration costs (if applicable)
   - Testing & deployment (exact amount and currency)
   - Annual maintenance (exact amount and currency)
   - Total amount with GST/taxes if mentioned
   - Additional features cost
   - Payment schedule if specified

6. **DEVELOPMENT & MAINTENANCE**
   - Platform-specific requirements (Google Play, App Store fees)
   - Domain and hosting details
   - Maintenance coverage and terms
   - Additional feature pricing model

7. **PROJECT DELIVERABLES**
   - List all deliverables mentioned in the template
   - Timeline if available in knowledge base

**FORMATTING RULES:**
- Use clear markdown formatting
- Maintain bullet points for features
- Bold section headers
- Preserve exact numbers and currency symbols
- Use tables for pricing if appropriate

**VALIDATION CHECKLIST BEFORE RESPONDING:**
✓ All features exist in retrieved context
✓ Pricing matches knowledge base exactly
✓ No invented technical specifications
✓ Project type accurately matched
✓ Currency format preserved
✓ Maintenance terms accurate

If the exact project type is not found, identify the CLOSEST match from the knowledge base and state:
"Based on your requirements, the closest match in our offerings is [Project Name]. Here's the proposal:"

If critical information is missing, explicitly state:
"The following information is not specified in our knowledge base: [list missing items]"

Generate the detailed proposal now:
"""

prompt = ChatPromptTemplate.from_template(template)

# Use GPT-4 for better accuracy
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=openai_api_key,
    max_tokens=4000
)

# ✅ Create RAG chain using LCEL (LangChain Expression Language)
# Function to format retrieved documents into a single text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to combine all input fields into one search query
def build_query(inputs: dict) -> str:
    query_parts = [
        f"Project Type: {inputs.get('project_type', '')}",
        f"Platforms: {inputs.get('platforms', '')}",
        f"Technology: {inputs.get('tech_type', '')}",
        f"Services: {inputs.get('services', '')}",
        f"Custom Prompt: {inputs.get('custom_prompt', '')}"
    ]
    return "\n".join(query_parts)

# Build the RAG chain properly (fixing the PyString error)
rag_chain = (
    {
        "context": RunnablePassthrough() | (lambda x: build_query(x)) | retriever | format_docs,
        "platforms": RunnablePassthrough(),
        "vendor_type": RunnablePassthrough(),
        "project_type": RunnablePassthrough(),
        "automations": RunnablePassthrough(),
        "ai_platforms": RunnablePassthrough(),
        "platform_category": RunnablePassthrough(),
        "tech_type": RunnablePassthrough(),
        "services": RunnablePassthrough(),
        "custom_prompt": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# FastAPI app setup
app = FastAPI(title="Smart Proposal Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProposalRequest(BaseModel):
    platforms: List[str] = []
    vendor_type: Optional[str] = None
    project_type: List[str] = []
    automations: List[str] = []
    ai_platforms: List[str] = []
    platform_category: Optional[str] = None
    tech_type: Optional[str] = None
    services: List[str] = []
    custom_prompt: str = ""

@app.post("/generate-proposal")
async def generate_proposal(request: ProposalRequest):
    """Generate a proposal based on user inputs from the frontend."""
    try:
        # Prepare inputs for the chain
        inputs = {
            "platforms": ", ".join(request.platforms) if request.platforms else "Not specified",
            "vendor_type": request.vendor_type or "Not specified",
            "project_type": ", ".join(request.project_type) if request.project_type else "Not specified",
            "automations": ", ".join(request.automations) if request.automations else "Not specified",
            "ai_platforms": ", ".join(request.ai_platforms) if request.ai_platforms else "Not specified",
            "platform_category": request.platform_category or "Not specified",
            "tech_type": request.tech_type or "Not specified",
            "services": ", ".join(request.services) if request.services else "Not specified",
            "custom_prompt": request.custom_prompt or "No additional requirements"
        }
        
        # Generate proposal using RAG chain
        proposal = rag_chain.invoke(inputs)
        
        return {
            "success": True,
            "proposal": proposal,
            "user_inputs": request.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")
    

@app.post("/ask")
async def ask_redirect(request: ProposalRequest):
    """Alias for /generate-proposal — allows frontend or n8n workflows using /ask."""
    return await generate_proposal(request)




@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "knowledge_base": "loaded",
        "documents_processed": len(splits)
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Smart Proposal Generator API",
        "version": "2.0",
        "endpoints": {
            "generate_proposal": "/generate-proposal (POST)",
            "health": "/health (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)