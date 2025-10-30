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
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader  # kept for fallback
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google API
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load environment
load_dotenv()

# ========== CONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")  # either JSON string or path to file
GOOGLE_DOC_ID = os.getenv("GOOGLE_DOC_ID")  # set to: 1NHtKV_D4QBdzLjOQ_n4EKgpu-upjFOCN1OjYWzYqmkA
CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "text-embedding-3-large"  # larger embedding for better accuracy
LLM_MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")

if not GOOGLE_DOC_ID:
    raise ValueError("Missing GOOGLE_DOC_ID in environment")

if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in environment (service account credentials)")
# Decode base64 if the JSON is encoded (Render env safe)
try:
    if GOOGLE_SERVICE_ACCOUNT_JSON.strip().startswith("{"):
        # it's already raw JSON
        pass
    else:
        import base64
        GOOGLE_SERVICE_ACCOUNT_JSON = base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON).decode("utf-8")
except Exception as e:
    raise ValueError(f"Failed to decode GOOGLE_SERVICE_ACCOUNT_JSON: {e}")


# ========== FASTAPI ==========
app = FastAPI(title="Proposal Generator - Google Doc KB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production: add your front-end domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Pydantic Models ==========
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

Return pure JSON with keys:
platforms, vendor_type, project_types, automations, platform_type, tech_stack, services, industry, ecommerce_type, budget_range.
Use null for missing values.
"""

proposal_prompt_template = """You are a proposal generation assistant. Use ONLY the provided context (from the knowledge base) to produce a professional proposal.

RULES (mandatory):
1. Do NOT invent facts.
2. Do NOT include placeholder phrases like "[Information not available in knowledge base]".
3. Do NOT add any extra closing paragraphs, conclusions, or marketing lines (e.g., "We look forward...").
4. If a required item is not present in the context, simply omit it (do not invent).
5. Keep the text formal and structured (sections and bullet lists allowed).

CONTEXT:
{context}

REQUIREMENTS:
{requirements}

Now generate the complete proposal below:
"""

intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=proposal_prompt_template, input_variables=["context", "requirements"])

# ========== Google Docs helpers ==========
def load_service_account_credentials(json_input: str):
    """
    Accepts either:
     - path to a service account JSON file, or
     - the full JSON as a string (recommended for Render env var)
    Returns google.oauth2.service_account.Credentials
    """
    try:
        # if it's a path to a file
        if os.path.exists(json_input):
            creds = service_account.Credentials.from_service_account_file(json_input, scopes=["https://www.googleapis.com/auth/documents.readonly"])
            return creds
        # else treat as JSON string
        info = json.loads(json_input)
        creds = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/documents.readonly"])
        return creds
    except Exception as e:
        raise RuntimeError(f"Failed to load service account credentials: {e}")

def fetch_google_doc_text(document_id: str, creds: service_account.Credentials) -> str:
    """
    Fetches document structure and converts it to plain text.
    It handles paragraphs, tables (rows/cols), and text runs.
    """
    service = build("docs", "v1", credentials=creds, cache_discovery=False)
    doc = service.documents().get(documentId=document_id).execute()
    body = doc.get("body", {})
    content = []

    def read_structural_elements(elements):
        for el in elements:
            if "paragraph" in el:
                parts = el["paragraph"].get("elements", [])
                para_text = []
                for p in parts:
                    text_run = p.get("textRun")
                    if text_run and "content" in text_run:
                        para_text.append(text_run["content"])
                if para_text:
                    content.append("".join(para_text).strip())
            elif "table" in el:
                table = el["table"]
                rows = table.get("tableRows", [])
                for row in rows:
                    cells = row.get("tableCells", [])
                    row_texts = []
                    for cell in cells:
                        cell_content = []
                        for c in cell.get("content", []):
                            # recursively read paragraphs inside cell
                            if "paragraph" in c:
                                parts = c["paragraph"].get("elements", [])
                                for p in parts:
                                    tr = p.get("textRun")
                                    if tr and "content" in tr:
                                        cell_content.append(tr["content"])
                        row_texts.append(" ".join([t.strip() for t in cell_content]).strip())
                    if row_texts:
                        content.append(" | ".join(row_texts))
            elif "sectionBreak" in el:
                # treat as paragraph separator
                content.append("\n")
            # other element types ignored
    read_structural_elements(body.get("content", []))
    # join with double newlines for splitting later
    return "\n\n".join([c for c in content if c])

# ========== Embeddings & Vectorstore ==========
# initialize embedding model (larger model for better accuracy)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def build_or_load_vectorstore_from_google_doc():
    """
    Loads Google Doc, splits text, embeds chunks, and persist to Chroma.
    If CHROMA_DIR already exists, reuse it to save time.
    """
    if os.path.exists(CHROMA_DIR):
        try:
            vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            print("Loaded existing Chroma DB.")
            return vs
        except Exception:
            # if loading fails, clear and rebuild
            shutil.rmtree(CHROMA_DIR)

    # load credentials and doc text
    creds = load_service_account_credentials(GOOGLE_SERVICE_ACCOUNT_JSON)
    print("Fetching Google Doc...")
    text = fetch_google_doc_text(GOOGLE_DOC_ID, creds)
    if not text or not text.strip():
        raise RuntimeError("Google Doc contains no text")

    # split to chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250, separators=["\n\n", "\n", ".", "!", "?"])
    # create a simple list of docs expected by Chroma
    texts = splitter.split_text(text)
    # create docs in LangChain style (list of dicts with page_content)
    docs = [{"page_content": t} for t in texts]

    print(f"Indexing {len(docs)} chunks into Chroma...")
    vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    # persist is automatic for recent chroma; keep call if supported
    try:
        vs.persist()
    except Exception:
        pass
    print("Chroma build complete.")
    return vs

# build vectorstore at startup (blocking)
vectorstore = build_or_load_vectorstore_from_google_doc()

# initialize LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# ========== Utilities ==========
def extract_intent_from_text(text: str) -> dict:
    try:
        prompt = intent_prompt.format(question=text)
        resp = llm.invoke(prompt)
        content = resp.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {}
    except Exception as e:
        print("Intent extraction failed:", e)
        return {}

def format_requirements(intent: dict, query: Query) -> str:
    parts = []
    if query.question:
        parts.append(f"**Original Request:** {query.question}")
    # prefer extracted intent keys
    for key in ["platform_type", "platforms", "vendor_type", "project_types", "automations", "tech_stack", "services", "industry", "ecommerce_type", "budget_range"]:
        val = intent.get(key)
        if val:
            if isinstance(val, list):
                parts.append(f"**{key.replace('_',' ').title()}:** {', '.join(val)}")
            else:
                parts.append(f"**{key.replace('_',' ').title()}:** {val}")
    # also include structured fields from the UI if they exist
    for k, v in query.dict().items():
        if k == "question" or not v:
            continue
        if isinstance(v, list):
            parts.append(f"**{k.replace('_',' ').title()}:** {', '.join(v)}")
        else:
            parts.append(f"**{k.replace('_',' ').title()}:** {v}")
    return "\n".join(parts)

def clean_generation(text: str) -> str:
    # remove placeholder bracketed phrases
    text = re.sub(r"\[Information[^\]]*\]", "", text)
    # remove any trailing Conclusion / Summary or marketing closing phrases
    text = re.sub(r"(?is)\n*\s*(#+\s*(Conclusion|Summary).*$|This proposal provides.*$|We look forward.*$|\bConclusion:.*$|\bSummary:.*$)", "", text)
    # trim repeated newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ========== Routes ==========
@app.get("/")
def root():
    return {"status": "online", "message": "Proposal Generator (Google Doc KB) is running", "endpoints": {"ask": "POST /ask"}}

@app.post("/ask", response_model=AskResponse)
def ask(query: Query):
    # build combined description from structured fields + question
    parts = []
    if query.question:
        parts.append(query.question.strip())
    for k, v in query.dict().items():
        if k == "question" or not v:
            continue
        if isinstance(v, list):
            parts.append(f"{k}: {', '.join(v)}")
        else:
            parts.append(f"{k}: {v}")
    combined_text = "\n\n".join(parts).strip()
    if not combined_text:
        raise HTTPException(status_code=400, detail="Provide question or selections in the request body")

    try:
        # extract intent
        intent = extract_intent_from_text(combined_text)

        # build requirements text for prompt
        requirements = format_requirements(intent, query)

        # retrieve relevant documents
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8, "score_threshold": 0.2})

        # use invoke on retriever (newer LangChain API)
        related_docs = retriever.invoke(combined_text)

        if not related_docs:
            # if no docs matched, return polite empty response (no hallucination)
            return {"answer": "No relevant information found in the knowledge base for the given request.", "extracted_intent": intent}

        context = "\n\n".join([d.page_content if hasattr(d, "page_content") else (d.get("page_content") if isinstance(d, dict) else str(d)) for d in related_docs])

        # prepare final prompt
        full_prompt = proposal_prompt.format(context=context, requirements=requirements)
        resp = llm.invoke(full_prompt)
        generated = resp.content or ""

        # clean unwanted endings and placeholders
        cleaned = clean_generation(generated)

        return {"answer": cleaned, "extracted_intent": intent}
    except Exception as e:
        print("Error in /ask:", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
