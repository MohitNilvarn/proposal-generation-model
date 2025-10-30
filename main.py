from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
import json

# Load environment variables
load_dotenv()

# ==========================
# CONFIGURATION
# ==========================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"
CHROMA_DIR = "chroma_store"

# ==========================
# FASTAPI INITIALIZATION
# ==========================
app = FastAPI(title="Smart Proposal Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# LOAD KNOWLEDGE BASE
# ==========================
print("📄 Loading knowledge base from", PDF_PATH)
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DIR)
vectorstore.persist()
print(f"✅ Loaded {len(splits)} chunks into Chroma.")

# ==========================
# PROMPTS
# ==========================
intent_extraction_prompt = """You are an expert at analyzing project requirements.
Extract structured information from the text below.

USER INPUT:
{question}

Return JSON with the following keys:
platforms, vendor_type, project_types, automations, platform_type, tech_stack, services, industry, ecommerce_type, budget_range.
If not found, return null for that field.
"""

proposal_prompt_template = """You are a professional proposal generator.
Use only information from the context to write a project proposal.

RULES:
1. Copy all prices, terms, and feature descriptions exactly as in context.
2. If missing info, write "[Information not available in knowledge base]".
3. Be concise and professional.

CONTEXT:
{context}

PROJECT REQUIREMENTS:
{requirements}

Now generate a full structured proposal:
- Project Overview
- Features (Admin, App, Website)
- Tech Stack
- Services Included
- Pricing Breakdown
- Maintenance/Support
- Timeline & Deliverables
"""

intent_prompt = PromptTemplate(template=intent_extraction_prompt, input_variables=["question"])
proposal_prompt = PromptTemplate(template=proposal_prompt_template, input_variables=["context", "requirements"])

# ==========================
# LLM INITIALIZATION
# ==========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

# ==========================
# MODELS
# ==========================
class Query(BaseModel):
    question: str | None = None
    industry: str | None = None
    ecommerce_type: str | None = None
    platforms: list[str] | None = None
    tech_focus: str | None = None
    service_type: str | None = None
    budget_range: str | None = None

class AskResponse(BaseModel):
    answer: str
    extracted_intent: dict | None = None

# ==========================
# UTILITIES
# ==========================
def extract_intent(question: str) -> dict:
    try:
        response = llm.invoke(intent_prompt.format(question=question))
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception as e:
        print(f"❌ Intent extraction failed: {e}")
        return {}

def format_requirements(intent: dict, query: Query) -> str:
    req = []
    if query.question:
        req.append(f"**User Request:** {query.question}")
    for key, val in intent.items():
        if val:
            req.append(f"**{key.replace('_', ' ').title()}:** {val}")
    # also include structured JSON fields (from frontend dropdowns)
    for field, val in query.dict().items():
        if field != "question" and val:
            req.append(f"**{field.replace('_', ' ').title()}:** {val}")
    return "\n".join(req)

# ==========================
# ROUTES
# ==========================
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Smart Proposal Generator is running",
        "endpoint": "POST /ask",
    }

@app.post("/ask", response_model=AskResponse)
async def ask(query: Query):
    try:
        if not query.question and not any(query.dict().values()):
            raise HTTPException(status_code=400, detail="Please provide a question or selections.")

        # combine all user data into a single text for intent extraction
        combined_text = query.question or ""
        structured_text = " ".join(
            [f"{k}: {v}" for k, v in query.dict().items() if v and k != "question"]
        )
        full_query_text = f"{combined_text}\n\n{structured_text}"

        # Extract intent
        intent = extract_intent(full_query_text)

        # Build requirements text
        requirements = format_requirements(intent, query)

        # Retrieve matching context from Chroma
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        related_docs = retriever.get_relevant_documents(full_query_text)
        if not related_docs:
            return {"answer": "[Information not available in knowledge base]", "extracted_intent": intent}

        context = "\n\n".join([doc.page_content for doc in related_docs])

        # Generate proposal
        prompt_text = proposal_prompt.format(context=context, requirements=requirements)
        response = llm.invoke(prompt_text)

        return {"answer": response.content, "extracted_intent": intent}

    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Proposal Generator Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
