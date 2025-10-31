from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
import json
import glob

# ---------------- CONFIG ----------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY")

KB_DIR = "Proposal"  # Folder containing all your .pdf, .docx, .txt, etc.

# ---------------- PROMPTS ----------------
intent_extraction_prompt = """You are an expert at extracting structured information from proposal requests.

Analyze the user's request and extract the following information:

**USER REQUEST:**
{question}

**TASK:**
Extract and identify key structured fields like:
- Platform(s)
- Vendor Type
- Project Type
- Automation Tools
- Platform Category
- Technology Stack
- Services Required

Return ONLY a JSON object like:
{{
  "platforms": ["Website", "Admin Panel"],
  "vendor_type": "Single Vendor",
  "project_types": ["Software Development"],
  "automations": ["CRM Automation"],
  "platform_type": "Property Listing",
  "tech_stack": ["Next Js"],
  "services": ["Website Development", "Maintenance"]
}}"""

structured_prompt_template = """You are a professional proposal generator.

**RULES:**
- COPY all pricing and details EXACTLY from the context.
- If info is missing, write: "[Information not available in knowledge base]"
- Do NOT calculate totals or invent prices.

**CONTEXT (Knowledge Base):**
{context}

**PROJECT REQUIREMENTS:**
{requirements}

Generate the proposal with sections:
1. Project Overview
2. Admin Panel Features
3. Website/App Features
4. Technology Stack
5. Services Included
6. Pricing Breakdown
7. Maintenance & Support
8. Timeline/Deliverables
"""

intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=structured_prompt_template, input_variables=["context", "requirements"])

# ---------------- LLM ----------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

# ---------------- LOAD DOCS KB ----------------
def load_documents_from_kb(kb_dir=KB_DIR):
    """Load all supported docs from KB folder"""
    print("📂 Loading knowledge base documents...")
    docs = []

    supported_files = []
    for ext in ["*.pdf", "*.txt", "*.md", "*.docx"]:
        supported_files.extend(glob.glob(os.path.join(kb_dir, ext)))

    if not supported_files:
        raise ValueError(f"No files found in {kb_dir}. Please add PDFs, DOCX, or TXT files.")

    for file in supported_files:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file)
        elif ext == ".txt" or ext == ".md":
            loader = TextLoader(file)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file)
        else:
            continue
        docs.extend(loader.load())

    print(f"✅ Loaded {len(docs)} documents from {len(supported_files)} files.")
    return docs

# Split and index
docs = load_documents_from_kb()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"✅ Created vectorstore with {len(chunks)} chunks.")

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Docs KB Proposal Generator (Local Vectorstore)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class Query(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    extracted_intent: dict = None

# ---------------- HELPERS ----------------
def extract_intent(question: str) -> dict:
    try:
        filled_prompt = intent_prompt.format(question=question)
        response = llm.invoke(filled_prompt)
        content = response.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except Exception as e:
        print(f"❌ Intent extraction error: {e}")
        return {}

def format_requirements_from_intent(intent: dict, question: str) -> str:
    lines = [f"**Original Request:** {question}"]
    if intent.get("platform_type"):
        lines.append(f"**Platform Type:** {intent['platform_type']}")
    if intent.get("platforms"):
        lines.append(f"**Platforms:** {', '.join(intent['platforms'])}")
    if intent.get("vendor_type"):
        lines.append(f"**Vendor Type:** {intent['vendor_type']}")
    if intent.get("project_types"):
        lines.append(f"**Project Types:** {', '.join(intent['project_types'])}")
    if intent.get("services"):
        lines.append(f"**Services:** {', '.join(intent['services'])}")
    return "\n".join(lines)

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Docs KB Proposal Generator is running",
        "knowledge_base_path": KB_DIR,
    }

@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    try:
        print(f"🟩 Received: {query.question}")
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Extract intent
        intent = extract_intent(query.question)
        requirements = format_requirements_from_intent(intent, query.question)

        # Search KB
        docs = vectorstore.similarity_search(query.question, k=5)
        if not docs:
            return {"answer": "[Information not available in knowledge base]", "extracted_intent": intent}

        context = "\n\n".join([d.page_content for d in docs])

        # Generate proposal
        filled_prompt = proposal_prompt.format(context=context, requirements=requirements)
        response = llm.invoke(filled_prompt)

        return {"answer": response.content, "extracted_intent": intent}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Proposal generation failed: {str(e)}")


@app.post("/reload-kb")
def reload_kb():
    """Reload all documents from the KnowledgeBase folder without restarting."""
    global vectorstore
    try:
        new_docs = load_documents_from_kb()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(new_docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return {"status": "success", "total_chunks": len(chunks), "message": "Knowledge base reloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
