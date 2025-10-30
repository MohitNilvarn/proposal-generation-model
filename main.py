# main.py
import os
import re
import json
import shutil
from dotenv import load_dotenv
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Google Docs + PDF support
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_community.document_loaders import PyPDFLoader

# Load environment
load_dotenv()

# ===== CONFIG =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_DOC_ID = os.getenv("GOOGLE_DOC_ID")
PDF_PATH = "Multi Vendor Food Delivery Platform - Kaamil 24 March 2025.pdf"
CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in environment")
if not GOOGLE_DOC_ID:
    raise ValueError("Missing GOOGLE_DOC_ID")

# ===== FASTAPI =====
app = FastAPI(title="Proposal Generator - Multi-Source KB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELS =====
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

# ===== PROMPTS =====
intent_extraction_prompt = """Extract structured info as JSON.
USER INPUT:
{question}
Return keys: platforms, project_types, services, industry, ecommerce_type, budget_range, tech_stack, automations.
"""
proposal_prompt_template = """You are a professional proposal writer.
Use ONLY the provided CONTEXT (from Google Doc & PDF) to create a structured, elaborated, and accurate proposal.

Do NOT invent or add fake info.
If something isn't found, skip it — do NOT guess.

CONTEXT:
{context}

REQUIREMENTS:
{requirements}

Generate a detailed yet precise proposal below:
"""

intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=proposal_prompt_template, input_variables=["context", "requirements"])

# ===== GOOGLE DOC HELPERS =====
def load_service_account_credentials(json_input: str):
    try:
        if os.path.exists(json_input):
            creds = service_account.Credentials.from_service_account_file(
                json_input, scopes=["https://www.googleapis.com/auth/documents.readonly"]
            )
        else:
            info = json.loads(json_input)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/documents.readonly"]
            )
        return creds
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Google credentials: {e}")

def fetch_google_doc_text(doc_id: str, creds) -> str:
    service = build("docs", "v1", credentials=creds, cache_discovery=False)
    doc = service.documents().get(documentId=doc_id).execute()
    text = []
    for el in doc.get("body", {}).get("content", []):
        if "paragraph" in el:
            for e in el["paragraph"].get("elements", []):
                tr = e.get("textRun")
                if tr and "content" in tr:
                    text.append(tr["content"])
    return "\n".join(text).strip()

# ===== VECTORSTORE =====
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def build_vectorstore():
    if os.path.exists(CHROMA_DIR):
        try:
            return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except:
            shutil.rmtree(CHROMA_DIR)

    creds = load_service_account_credentials(GOOGLE_SERVICE_ACCOUNT_JSON)
    print("📄 Fetching Google Doc...")
    google_text = fetch_google_doc_text(GOOGLE_DOC_ID, creds)
    google_chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250).split_text(google_text)
    google_docs = [Document(page_content=t, metadata={"source": "google_doc"}) for t in google_chunks]

    print("📘 Loading PDF...")
    pdf_loader = PyPDFLoader(PDF_PATH)
    pdf_docs = pdf_loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250))
    for d in pdf_docs:
        d.metadata["source"] = "pdf_proposal"

    all_docs = google_docs + pdf_docs
    print(f"🧠 Total chunks indexed: {len(all_docs)}")

    vs = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    try: vs.persist()
    except: pass
    return vs

vectorstore = build_vectorstore()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# ===== UTILITIES =====
def extract_intent(text: str) -> dict:
    try:
        resp = llm.invoke(intent_prompt.format(question=text))
        match = re.search(r"\{.*\}", resp.content, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except:
        return {}

def format_requirements(intent, query):
    lines = []
    if query.question:
        lines.append(f"**Request:** {query.question}")
    for k, v in {**intent, **query.dict()}.items():
        if v and k != "question":
            val = ", ".join(v) if isinstance(v, list) else v
            lines.append(f"**{k.replace('_', ' ').title()}:** {val}")
    return "\n".join(lines)

def clean_output(text):
    text = re.sub(r"(?i)(we look forward.*|thank you.*)$", "", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ===== ROUTES =====
@app.get("/")
def root():
    return {"status": "online", "message": "Proposal Agent is live!"}

@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    text = "\n".join([f"{k}: {v}" for k, v in query.dict().items() if v])
    if not text:
        raise HTTPException(400, "Provide question or structured inputs")

    intent = extract_intent(text)
    reqs = format_requirements(intent, query)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(text)
    context = "\n\n".join([d.page_content for d in docs])

    resp = llm.invoke(proposal_prompt.format(context=context, requirements=reqs))
    return {"answer": clean_output(resp.content), "extracted_intent": intent}
