from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import PromptTemplate

from fastapi.middleware.cors import CORSMiddleware
from supabase.client import create_client
import os
from fastapi import File, UploadFile
import shutil
import uuid
from dotenv import load_dotenv
import re

load_dotenv()

# Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL", "https://hjofholmamrathafloam.supabase.co")
supabase_key = os.getenv("SUPABASE_KEY")

if not openai_api_key or not supabase_key:
    raise ValueError("Missing required environment variables: OPENAI_API_KEY or SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(supabase_url, supabase_key)

# PDF Path
PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"

# Intent extraction prompt
intent_extraction_prompt = """You are an expert at extracting structured information from proposal requests.

Analyze the user's request and extract the following information:

**USER REQUEST:**
{question}

**TASK:**
Extract and identify:

1. **Platforms**: Which of these are mentioned or implied?
   - Landing Page Website
   - Fully Functional Website
   - Admin Panel
   - Android App
   - iOS App
   - Desktop App

2. **Vendor Type**: Is it Single Vendor or Multi Vendor?

3. **Project Types**: Which apply?
   - Software Development
   - AI Automations

4. **Automations**: Any of these mentioned?
   - Manychats
   - CRM Automation
   - AI Calling

5. **Platform Type**: What kind of platform?
   - Property Listing
   - Property Selling & Rental
   - Service Provider
   - Ecommerce
   - Sales Management
   - CRM
   - Car Listing Platform
   - Bidding Platform
   - Courses Selling
   - Seminar Registration
   - Matrimony
   - Dating Platform

6. **Technology**: Any specific tech mentioned?
   - Shopify
   - Next Js
   - Wordpress

7. **Services**: Which services are mentioned or implied?
   - UI UX Designs
   - Website Development
   - App Development
   - Testing
   - Maintenance
   - Deployment

Return ONLY a JSON object with the extracted information. Use null for missing information.

Example format:
{{
  "platforms": ["Fully Functional Website", "Admin Panel"],
  "vendor_type": "Single Vendor",
  "project_types": ["Software Development"],
  "automations": null,
  "platform_type": "Property Listing",
  "tech_stack": null,
  "services": ["UI UX Designs", "Website Development"]
}}"""

# Enhanced Prompt Template
structured_prompt_template = """You are a proposal generation assistant. Generate proposals ONLY using exact text from the context provided.

**CRITICAL RULES - DO NOT BREAK:**
1. COPY pricing amounts EXACTLY as they appear in the context. Do not change, round, or modify numbers.
2. COPY all text verbatim from the context. Do not paraphrase or rewrite.
3. If information is missing, write: "[Information not available in knowledge base]"
4. Do not invent features, pricing, or any details.
5. Do not calculate totals or modify prices.
6. Match the components to what exists in the knowledge base.

**CONTEXT (USE EXACTLY AS PROVIDED):**
{context}

**PROJECT REQUIREMENTS:**
{requirements}

**TASK:**
Based on the project requirements above, extract and present the exact information from the context that matches these specifications.

Generate a comprehensive proposal with the following sections:

1. **Project Overview**
   - Platform type and description
   - Vendor model (if applicable)
   - Platforms to be developed

2. **Admin Panel Features** (if Admin Panel is requested)
   - Copy exact features from knowledge base for this platform type

3. **Website/App Features** (if Website/App is requested)
   - Copy exact features from knowledge base for this platform type

4. **Technology Stack**
   - Technologies mentioned in knowledge base for this type of project

5. **Services Included**
   - Detail each requested service using information from knowledge base

6. **Pricing Breakdown**
   - Copy EXACT pricing for each component as it appears in knowledge base
   - Do NOT calculate totals or combine prices
   - Show pricing exactly as written

7. **Maintenance & Support** (if included in services)
   - Copy exact maintenance terms from knowledge base

8. **Timeline/Deliverables** (if available)

**CRITICAL REMINDERS:**
- Only use information that exists in the context
- Copy all pricing, features, and specifications word-for-word
- If the exact combination isn't in knowledge base, use the closest match
- Never modify numbers or calculate totals
- Format as a professional proposal document"""

intent_prompt = PromptTemplate(
    template=intent_extraction_prompt,
    input_variables=["question"]
)

proposal_prompt = PromptTemplate(
    template=structured_prompt_template,
    input_variables=["context", "requirements"]
)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_api_key
)

# Initialize FastAPI app
app = FastAPI(title="Proposal Generator API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Remove this in production and specify exact origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class UploadRequest(BaseModel):
    pdf_path: str = PDF_PATH

class Query(BaseModel):
    question: str

class UploadResponse(BaseModel):
    status: str
    total_chunks: int
    message: str

class AskResponse(BaseModel):
    answer: str
    extracted_intent: dict = None

# Global embeddings instance
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=openai_api_key
)

def extract_intent(question: str) -> dict:
    """Extract structured intent from free-form question using LLM"""
    try:
        filled_prompt = intent_prompt.format(question=question)
        response = llm.invoke(filled_prompt)
        
        # Extract JSON from response
        content = response.content.strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            import json
            intent_data = json.loads(json_match.group())
            return intent_data
        
        return {}
    except Exception as e:
        print(f"Error extracting intent: {e}")
        return {}

def build_search_query_from_intent(intent: dict, original_question: str) -> str:
    """Build optimized search query from extracted intent"""
    query_parts = []
    
    # Add platform type (most important)
    if intent.get("platform_type"):
        query_parts.append(intent["platform_type"])
    
    # Add vendor type
    if intent.get("vendor_type"):
        query_parts.append(intent["vendor_type"])
    
    # Add key platforms
    platforms = intent.get("platforms", [])
    if platforms:
        if "Admin Panel" in platforms:
            query_parts.append("admin panel")
        if any("Website" in p for p in platforms):
            query_parts.append("website")
        if any("App" in p for p in platforms):
            query_parts.append("mobile app")
    
    # Add project types
    project_types = intent.get("project_types", [])
    if project_types:
        query_parts.extend(project_types)
    
    # Add automations
    if intent.get("automations"):
        query_parts.append(intent["automations"])
    
    # Add tech stack
    if intent.get("tech_stack"):
        query_parts.append(intent["tech_stack"])
    
    # Add top services
    services = intent.get("services", [])
    if services:
        query_parts.extend(services[:2])  # Top 2 services
    
    # If we couldn't extract much, fall back to original question
    if len(query_parts) < 2:
        return original_question
    
    return " ".join(query_parts)

def format_requirements_from_intent(intent: dict, original_question: str) -> str:
    """Format extracted intent into readable requirements"""
    requirements = []
    
    requirements.append(f"**Original Request:** {original_question}\n")
    
    if intent.get("platform_type"):
        requirements.append(f"**Platform Type:** {intent['platform_type']}")
    
    platforms = intent.get("platforms", [])
    if platforms:
        requirements.append(f"**Platforms Required:** {', '.join(platforms)}")
    
    if intent.get("vendor_type"):
        requirements.append(f"**Vendor Model:** {intent['vendor_type']}")
    
    project_types = intent.get("project_types", [])
    if project_types:
        requirements.append(f"**Project Types:** {', '.join(project_types)}")
    
    if intent.get("automations"):
        requirements.append(f"**Automations:** {intent['automations']}")
    
    if intent.get("tech_stack"):
        requirements.append(f"**Technology Preference:** {intent['tech_stack']}")
    
    services = intent.get("services", [])
    if services:
        requirements.append(f"**Services to Include:** {', '.join(services)}")
    
    return "\n".join(requirements)

# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Proposal Generator API is running",
        "endpoints": {
            "upload-file": "POST /upload-file",
            "ask": "POST /ask"
        }
    }

@app.post("/upload-file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported")

        # Save uploaded file to a temporary location accessible by backend
        temp_filename = f"uploads/{uuid.uuid4().hex}_{file.filename}"
        os.makedirs(os.path.dirname(temp_filename), exist_ok=True)

        with open(temp_filename, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Reuse your upload logic: load temp_filename
        loader = PyPDFLoader(temp_filename)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        splits = text_splitter.split_documents(docs)

        vector_db = SupabaseVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            client=supabase,
            table_name="embeddings",
            chunk_size=5
        )

        # Clean up temp file
        try:
            os.remove(temp_filename)
        except:
            pass

        return {
            "status": "success",
            "total_chunks": len(splits),
            "message": f"Uploaded {len(splits)} chunks successfully"
        }

    except Exception as e:
        print("Error in upload_file:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced query endpoint with intent extraction
@app.post("/ask", response_model=AskResponse)
async def ask(query: Query):
    try:
        print(f"Received query: {query.question}")
        
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Step 1: Extract structured intent from the question
        print("Extracting intent from question...")
        intent = extract_intent(query.question)
        print(f"Extracted intent: {intent}")
        
        # Step 2: Build optimized search query
        search_query = build_search_query_from_intent(intent, query.question)
        print(f"Search query: {search_query}")
        
        # Step 3: Create query embedding
        query_embedding = embeddings.embed_query(search_query)
        
        # Step 4: Retrieve similar documents from Supabase
        results = supabase.rpc(
            "match_embeddings",
            {
                "query_embedding": query_embedding,
                "match_count": 10,  # Get more results for better coverage
                "match_threshold": 0.0
            }
        ).execute()
        
        if not results.data:
            return {
                "answer": "[Information not available in knowledge base]. Please ensure the PDF has been uploaded using the /upload-file endpoint.",
                "extracted_intent": intent
            }
        
        retrieved_docs = results.data
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Step 5: Prepare context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Step 6: Format requirements from intent
        requirements = format_requirements_from_intent(intent, query.question)
        
        # Step 7: Generate proposal using LLM
        filled_prompt = proposal_prompt.format(context=context, requirements=requirements)
        response = llm.invoke(filled_prompt)
        
        print("Generated proposal successfully")
        
        return {
            "answer": response.content,
            "extracted_intent": intent
        }
        
    except Exception as e:
        print(f"Error in ask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print(f"PDF Path: {PDF_PATH}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)