import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"
CHROMA_PERSIST_DIR = "chroma_db"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
print(f"Created {len(splits)} chunks")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# ---- Corrected: pass embedding= instead of embedding_function ----
vector_db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)
vector_db.persist()
print("Chroma DB built and persisted to", CHROMA_PERSIST_DIR)
