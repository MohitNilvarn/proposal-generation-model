# main.py
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

# ✅ Correct imports for new LangChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # ✅ FIXED import

# Google API imports
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

# ========== Prompts ==========
intent_extraction_prompt = """You are an expert at extracting structured information from user requests.

USER INPUT:
{question}

Return JSON with keys:
platforms, vendor_type, project_types, automations, platform_type, tech_stack, services, industry, ecommerce_type, budget_range.
Use null for missing values.
"""

proposal_prompt_template = """You are a proposal generation assistant. Use ONLY the provided context to produce a professional project proposal.

Rules:
1. Do NOT invent facts.
2. Do NOT include placeholder phrases like "[Information not available in knowledge base]".
3. Do NOT add extra marketing or closing statements.
4. Omit missing details — do not guess or fabricate.
5. Keep it clean, professional, and structured.

CONTEXT:
{context}

REQUIREMENTS:
{requirements}

Now generate the final proposal:
"""

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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
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
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# ========== Utils ==========
def extract_intent_from_text(text: str) -> dict:
    try:
        resp = llm.invoke(intent_prompt.format(question=text))
        content = resp.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}

def format_requirements(intent: dict, query: Query) -> str:
    parts = []
    if query.question:
        parts.append(f"**Original Request:** {query.question}")
    for key, val in intent.items():
        if val:
            if isinstance(val, list):
                parts.append(f"**{key.replace('_',' ').title()}:** {', '.join(val)}")
            else:
                parts.append(f"**{key.replace('_',' ').title()}:** {val}")
    for k, v in query.dict().items():
        if k == "question" or not v:
            continue
        parts.append(f"**{k.replace('_',' ').title()}:** {', '.join(v) if isinstance(v, list) else v}")
    return "\n".join(parts)

def clean_generation(text: str) -> str:
    text = re.sub(r"\[Information[^\]]*\]", "", text)
    text = re.sub(r"(?is)\n*\s*(#+\s*(Conclusion|Summary).*$|We look forward.*$|This proposal provides.*$)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ========== Routes ==========
@app.get("/")
def root():
    return {"status": "online", "message": "✅ Proposal Generator ready", "endpoints": ["/ask"]}

@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    combined_text = "\n".join([query.question or ""] + [f"{k}: {v}" for k, v in query.dict().items() if v and k != "question"])
    if not combined_text.strip():
        raise HTTPException(status_code=400, detail="Provide a question or structured input")

    try:
        intent = extract_intent_from_text(combined_text)
        requirements = format_requirements(intent, query)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        related_docs = retriever.invoke(combined_text)

        if not related_docs:
            return {"answer": "No relevant information found in the knowledge base.", "extracted_intent": intent}

        context = "\n\n".join(d.page_content for d in related_docs)
        prompt = proposal_prompt.format(context=context, requirements=requirements)
        resp = llm.invoke(prompt)
        return {"answer": clean_generation(resp.content or ""), "extracted_intent": intent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
