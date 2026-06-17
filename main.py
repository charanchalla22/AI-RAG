import os
import shutil
import asyncpg  
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

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

MODEL = ChatGroq(model="qwen/qwen3-32b", reasoning_format="parsed")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Database DSN (string)
DB_DSN = "postgresql://postgres:charan@localhost:5432/postgres"
VECTOR_STORE = None
AGENT = None

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── NEW HISTORY ENDPOINT (Keeps your frontend from throwing 404s on New Session) ───
@app.get("/history")
async def get_chat_history(session_id: str):
    """Retrieves chronological database entries to rebuild UI chat states."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(
            "SELECT sender, message_text FROM chat_messages WHERE session_id = $1 ORDER BY created_at ASC",
            session_id
        )
        if not rows:
            return {"status": "success", "history": []}
        
        formatted_history = []
        for r in rows:
            role = "user" if r["sender"] == "user" else "assistant"
            formatted_history.append({"role": role, "content": r["message_text"]})
        return {"status": "success", "history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global VECTOR_STORE, AGENT
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 1️⃣ FILTER OUT NON-TEXT / IMAGE-ONLY SCANNED PDFS
        total_text = "".join([doc.page_content for doc in docs]).strip()
        if not total_text or len(total_text) < 10:  # Adjust character threshold if needed
            raise HTTPException(
                status_code=400, 
                detail="This PDF contains no extractable text. Scanned images or blank files are not supported."
            )
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        VECTOR_STORE = InMemoryVectorStore.from_documents(all_splits, EMBEDDINGS)
        
        @tool
        def retrieve_context(query: str) -> str:
            """Retrieves relevant context from the PDF document based on the query."""
            similar_docs = VECTOR_STORE.similarity_search(query, k=3)
            
            # 2️⃣ DETECT ABSENCE OF RELATED CONTEXT DURING RETRIEVAL
            if not similar_docs:
                return "CRITICAL ERROR: No relevant matches or text contexts could be extracted from the uploaded document for this query topic."
            
            data = []
            for doc in similar_docs:
                content = doc.page_content
                source = doc.metadata.get("source", "unknown")
                data.append(f"Content: {content}\nSource: {source}")
            return "\n\n".join(data)
    
        tools = [retrieve_context]
        
        # 3️⃣ STRICT SYSTEM PROMPT GUARDRAIL AGAINST UNRELATED CONTENT
        client_prompt = (
            "You are an assistant that exclusively answers queries using information retrieved from the uploaded PDF document.\n"
            "CRITICAL RULES:\n"
            "1. You must call the `retrieve_context` tool for every query to find answers.\n"
            "2. If the user asks a question about a completely unrelated topic, or if the retrieved context does not "
            "contain the answer, you MUST respond with: 'I am sorry, but the provided document does not contain information related to this topic.'\n"
            "3. Do not make up facts or use external knowledge outside of the document context."
        )
        
        AGENT = create_agent(MODEL, tools, system_prompt=client_prompt)
        
        return {"status": "success", "message": f"Successfully parsed and indexed: {file.filename}"}
        
    except HTTPException:
        # Re-raise explicit HTTP exceptions to prevent them from being swallowed by general catch-all block
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),      
    session_id: str = Form(...)    
):
    global AGENT
    if not AGENT:
        raise HTTPException(status_code=400, detail="No active document matrix found. Upload a file on the sidebar.")
        
    conn = await asyncpg.connect(DB_DSN)
    
    try:
        await conn.execute(
            """
            INSERT INTO chat_sessions (session_id, user_id) 
            VALUES ($1, $2) 
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id, user_id
        )
        
        await conn.execute(
            "INSERT INTO chat_messages (session_id, sender, message_text) VALUES ($1, $2, $3)",
            session_id, "user", message
        )
        
        rows = await conn.fetch(
            "SELECT sender, message_text FROM chat_messages WHERE session_id = $1 ORDER BY created_at ASC",
            session_id
        )
        
        formatted_history = []
        for r in rows:
            role = "user" if r["sender"] == "user" else "assistant"
            formatted_history.append({"role": role, "content": r["message_text"]})
            
        response = AGENT.invoke({"messages": formatted_history})
        ai_message = response["messages"][-1].content
        
        await conn.execute(
            "INSERT INTO chat_messages (session_id, sender, message_text) VALUES ($1, $2, $3)",
            session_id, "ai", ai_message
        )
        
        return {"response": ai_message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()