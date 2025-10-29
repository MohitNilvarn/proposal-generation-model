from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import re
import shutil

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment")

# File path to your knowledge base PDF
PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"
CHROMA_PATH = "chroma_store"

# ===== PROMPTS =====
intent_extraction_prompt = """
You are an expert at analyzing project requests and extracting structured information.

USER REQUEST:
{question}

Extract and identify:

1. Platforms (e.g., Website, Admin Panel, App)
2. Vendor Type (Single/Multi Vendor)
3. Project Type (Software Development, AI Automations)
4. Platform Type (e.g., Property Listing, Ecommerce, CRM)
5. Technology (e.g., Next.js, Shopify, Wordpress)
6. Services (e.g., UI/UX Design, Development, Maintenance)

Return ONLY valid JSON:
{
  "platforms": ["..."],
  "vendor_type": "...",
  "project_types": ["..."],
  "platform_type": "...",
  "tech_stack": "...",
  "services": ["..."]
}
"""

structured_prompt_template = """
You are a proposal generation assistant. Use only the provided context below.

RULES:
1. Copy all pricing, features, and text exactly from the context.
2. Do not paraphrase, modify, or calculate totals.
3. If something is missing, write "[Information not available in knowledge base]".

CONTEXT:
{context}

REQUIREMENTS:
{requirements}

TASK:
Generate a structured proposal document with:

1. Project Overview
2. Admin Panel Features (if applicable)
3. Website/App Features (if applicable)
4. Technology Stack
5. Services Included
6. Pricing Breakdown
7. Maintenance & Support
8. Timeline/Deliverables
"""

# ===== MODELS =====
class Query(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    extracted_intent: dict = None

# ===== FastAPI App =====
app = FastAPI(title="Proposal Generator (No Supabase)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Initialize LLM and Embeddings =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=structured_prompt_template, input_variables=["context", "requirements"])

# ===== VectorStore Initialization =====
def load_pdf_into_chroma():
    """Load local PDF into Chroma vectorstore"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    print(f"📄 Loading knowledge base from {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=CHROMA_PATH)
    vectorstore.persist()
    print(f"✅ Loaded {len(splits)} chunks into Chroma.")
    return vectorstore

vectorstore = None

@app.on_event("startup")
async def startup_event():
    global vectorstore
    vectorstore = load_pdf_into_chroma()

# ===== Utility Functions =====
def extract_intent(question: str) -> dict:
    """Extract structured intent using LLM"""
    try:
        filled_prompt = intent_prompt.format(question=question)
        response = llm.invoke(filled_prompt)
        content = response.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            import json
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        print(f"❌ Intent extraction failed: {e}")
        return {}

def format_requirements(intent: dict, question: str) -> str:
    """Readable summary of user requirements"""
    lines = [f"**User Request:** {question}"]
    for k, v in intent.items():
        if v:
            lines.append(f"**{k.replace('_', ' ').title()}:** {v}")
    return "\n".join(lines)

# ===== Routes =====
@app.get("/")
async def root():
    return {"status": "online", "message": "Proposal Generator running", "endpoint": "POST /ask"}

@app.post("/ask", response_model=AskResponse)
async def ask(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Extract intent
    intent = extract_intent(query.question)
    print("🧠 Extracted intent:", intent)

    # Create search query (combine important intent keywords)
    search_query = " ".join(
        str(v) for v in intent.values() if v
    ) or query.question

    # Retrieve relevant chunks from Chroma
    results = vectorstore.similarity_search(search_query, k=5)
    if not results:
        return {"answer": "[Information not available in knowledge base]", "extracted_intent": intent}

    context = "\n\n".join([doc.page_content for doc in results])
    requirements = format_requirements(intent, query.question)

    # Generate proposal
    filled_prompt = proposal_prompt.format(context=context, requirements=requirements)
    response = llm.invoke(filled_prompt)

    return {"answer": response.content, "extracted_intent": intent}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
