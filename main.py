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

# LangChain / OpenAI imports
try:
    from langchain.schema import Document  # older versions
except ImportError:
    from langchain_core.documents import Document  # newer versions

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google API
from google.oauth2 import service_account
from googleapiclient.discovery import build
import base64

# Load environment
load_dotenv()

# ===================== CONFIG =====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_DOC_ID = os.getenv("GOOGLE_DOC_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SERVICE_ACCOUNT_B64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_B64")  # Base64 version for Render
CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY")

if not GOOGLE_DOC_ID:
    raise ValueError("❌ Missing GOOGLE_DOC_ID")

# Prefer base64 if available
if GOOGLE_SERVICE_ACCOUNT_B64 and not GOOGLE_SERVICE_ACCOUNT_JSON:
    try:
        GOOGLE_SERVICE_ACCOUNT_JSON = base64.b64decode(GOOGLE_SERVICE_ACCOUNT_B64).decode("utf-8")
        print("✅ Loaded Google credentials from base64 env var")
    except Exception as e:
        raise ValueError(f"❌ Failed to decode GOOGLE_SERVICE_ACCOUNT_B64: {e}")

if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise ValueError("❌ Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_B64")

# ===================== FASTAPI =====================
app = FastAPI(title="Proposal Generator (Google Doc KB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, limit this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MODELS =====================
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

# ===================== PROMPTS =====================
intent_extraction_prompt = """You are an expert at extracting structured information from user requests.

USER INPUT:
{question}

Return pure JSON with keys:
platforms, vendor_type, project_types, automations, platform_type, tech_stack, services, industry, ecommerce_type, budget_range.
Use null for missing values.
"""

proposal_prompt_template = """You are a proposal generation assistant. Use ONLY the provided context (from the knowledge base) to produce a professional proposal.

RULES:
1. Do NOT invent facts.
2. Do NOT include placeholder phrases like "[Information not available]".
3. Do NOT add extra closing paragraphs or conclusions.
4. If a field is missing, omit it entirely.

CONTEXT:
{context}

REQUIREMENTS:
{requirements}

Now generate the proposal below:
"""

intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=proposal_prompt_template, input_variables=["context", "requirements"])

# ===================== GOOGLE DOC HELPERS =====================
def load_service_account_credentials(json_input: str):
    """Load Google service account credentials from JSON string or file path."""
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

def fetch_google_doc_text(document_id: str, creds: service_account.Credentials) -> str:
    """Fetches a Google Doc and converts it to plain text."""
    service = build("docs", "v1", credentials=creds, cache_discovery=False)
    doc = service.documents().get(documentId=document_id).execute()
    body = doc.get("body", {})
    content = []

    def read_elements(elements):
        for el in elements:
            if "paragraph" in el:
                parts = el["paragraph"].get("elements", [])
                text = "".join(
                    p.get("textRun", {}).get("content", "") for p in parts if "textRun" in p
                ).strip()
                if text:
                    content.append(text)
            elif "table" in el:
                for row in el["table"].get("tableRows", []):
                    row_texts = []
                    for cell in row.get("tableCells", []):
                        cell_texts = []
                        for c in cell.get("content", []):
                            if "paragraph" in c:
                                for p in c["paragraph"].get("elements", []):
                                    if "textRun" in p:
                                        cell_texts.append(p["textRun"]["content"].strip())
                        row_texts.append(" ".join(cell_texts))
                    content.append(" | ".join(row_texts))
            elif "sectionBreak" in el:
                content.append("\n")

    read_elements(body.get("content", []))
    return "\n\n".join([c for c in content if c])

# ===================== VECTOR STORE =====================
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def build_or_load_vectorstore_from_google_doc():
    """Fetch doc, chunk text, embed, and persist in Chroma."""
    if os.path.exists(CHROMA_DIR):
        try:
            print("✅ Loading existing Chroma...")
            return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception:
            shutil.rmtree(CHROMA_DIR)

    creds = load_service_account_credentials(GOOGLE_SERVICE_ACCOUNT_JSON)
    print("📄 Fetching Google Doc content...")
    text = fetch_google_doc_text(GOOGLE_DOC_ID, creds)

    if not text.strip():
        raise RuntimeError("❌ Google Doc is empty")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    print(f"🧠 Indexing {len(docs)} chunks...")
    vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    try:
        vs.persist()
    except Exception:
        pass
    print("✅ Chroma build complete")
    return vs

vectorstore = build_or_load_vectorstore_from_google_doc()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# ===================== HELPERS =====================
def extract_intent_from_text(text: str) -> dict:
    try:
        prompt = intent_prompt.format(question=text)
        resp = llm.invoke(prompt)
        content = resp.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}

def format_requirements(intent: dict, query: Query) -> str:
    parts = []
    if query.question:
        parts.append(f"**Original Request:** {query.question}")
    for k, v in {**intent, **query.dict()}.items():
        if v and k != "question":
            val = ", ".join(v) if isinstance(v, list) else v
            parts.append(f"**{k.replace('_',' ').title()}:** {val}")
    return "\n".join(parts)

def clean_generation(text: str) -> str:
    text = re.sub(r"\[Information[^\]]*\]", "", text)
    text = re.sub(r"(?is)\n*\s*(#+\s*(Conclusion|Summary).*$|We look forward.*$)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ===================== ROUTES =====================
@app.get("/")
def root():
    return {"status": "online", "message": "Proposal Generator running"}

@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    combined = "\n".join([f"{k}: {v}" for k, v in query.dict().items() if v])
    if not combined:
        raise HTTPException(status_code=400, detail="Missing question or input fields")

    try:
        intent = extract_intent_from_text(combined)
        requirements = format_requirements(intent, query)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        related_docs = retriever.invoke(combined)
        if not related_docs:
            return {"answer": "No relevant information found.", "extracted_intent": intent}

        context = "\n\n".join([d.page_content for d in related_docs])
        prompt = proposal_prompt.format(context=context, requirements=requirements)
        resp = llm.invoke(prompt)

        return {"answer": clean_generation(resp.content), "extracted_intent": intent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
