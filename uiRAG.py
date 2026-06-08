import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool 
from langchain.agents import create_agent 

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Shared memory spaces
MODEL = ChatGroq(model="qwen/qwen3-32b", reasoning_format="parsed")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE = None
AGENT = None
CHAT_HISTORY = []

# Define standard storage path inside the new backend/documents folder
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global VECTOR_STORE, AGENT, CHAT_HISTORY
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save file inside our dedicated backend/documents folder
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        VECTOR_STORE = InMemoryVectorStore.from_documents(all_splits, EMBEDDINGS)
        
        @tool
        def retrieve_context(query: str) -> str:
            """Retrieves relevant context from the PDF document based on the query."""
            similar_docs = VECTOR_STORE.similarity_search(query, k=3)
            data = []
            for doc in similar_docs:
                content = doc.page_content
                source = doc.metadata.get("source", "unknown")
                data.append(f"Content: {content}\nSource: {source}")
            return "\n\n".join(data)
        
        tools = [retrieve_context]
        client_prompt = "You are an agent who retrieves context from PDF docs."
        AGENT = create_agent(MODEL, tools, system_prompt=client_prompt)
        CHAT_HISTORY = []
        
        return {"status": "success", "message": f"Successfully parsed and indexed: {file.filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up file after ingestion
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat")
async def chat(message: str = Form(...)):
    global AGENT, CHAT_HISTORY
    if not AGENT:
        raise HTTPException(status_code=400, detail="No active document matrix found. Upload a file on the sidebar.")
        
    try:
        CHAT_HISTORY.append({"role": "user", "content": message})
        response = AGENT.invoke({"messages": CHAT_HISTORY})
        ai_message = response["messages"][-1].content
        CHAT_HISTORY.append({"role": "assistant", "content": ai_message})
        return {"response": ai_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))