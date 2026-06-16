import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# MODELS
# ==================================================

MODEL = ChatGroq(
    model="qwen/qwen3-32b"
)

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__),
    "documents"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================================================
# DATABASE
# ==================================================

def get_db():
    conn = psycopg2.connect(
        os.getenv("DATABASE_URL"),
        cursor_factory=RealDictCursor
    )

    try:
        yield conn
    finally:
        conn.close()

# ==================================================
# STARTUP
# ==================================================

@app.on_event("startup")
def startup():

    conn = psycopg2.connect(
        os.getenv("DATABASE_URL")
    )

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks(
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255),
        content TEXT,
        source VARCHAR(255)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) UNIQUE,
        user_id VARCHAR(255),
        title VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history(
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255),
        user_id VARCHAR(255),
        role VARCHAR(50),
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

# ==================================================
# HOME
# ==================================================

@app.get("/")
async def home():
    return FileResponse(
        os.path.join(
            os.path.dirname(__file__),
            "../templates/index.html"
        )
    )

# ==================================================
# CREATE NEW SESSION
# ==================================================

@app.get("/new-session")
async def new_session():
    return {
        "session_id": str(uuid.uuid4())
    }

# ==================================================
# LIST ALL SESSIONS
# ==================================================

@app.get("/sessions/{user_id}")
async def get_sessions(
    user_id: str,
    db=Depends(get_db)
):
    cur = db.cursor()

    cur.execute("""
    SELECT session_id,title
    FROM conversations
    WHERE user_id=%s
    ORDER BY created_at DESC
    """,(user_id,))

    return cur.fetchall()

# ==================================================
# PDF UPLOAD
# ==================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db=Depends(get_db)
):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{user_id}_{file.filename}"
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(
            docs
        )

        cur = db.cursor()

        cur.execute(
            "DELETE FROM document_chunks WHERE user_id=%s",
            (user_id,)
        )

        for chunk in chunks:

            cur.execute(
                """
                INSERT INTO document_chunks
                (user_id,content,source)
                VALUES (%s,%s,%s)
                """,
                (
                    user_id,
                    chunk.page_content,
                    file.filename
                )
            )

        db.commit()

        return {
            "message":
            f"{file.filename} uploaded successfully"
        }

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)

# ==================================================
# CHAT
# ==================================================

@app.post("/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),
    session_id: str = Form(...),
    db=Depends(get_db)
):

    try:

        cur = db.cursor()

        cur.execute("""
        INSERT INTO conversations
        (session_id,user_id,title)
        VALUES(%s,%s,%s)
        ON CONFLICT(session_id)
        DO NOTHING
        """,
        (
            session_id,
            user_id,
            message[:50]
        ))

        cur.execute(
            """
            SELECT content,source
            FROM document_chunks
            WHERE user_id=%s
            """,
            (user_id,)
        )

        rows = cur.fetchall()

        if not rows:
            raise HTTPException(
                status_code=400,
                detail="Upload PDF first"
            )

        docs = []

        for row in rows:
            docs.append(
                Document(
                    page_content=row["content"],
                    metadata={
                        "source":
                        row["source"]
                    }
                )
            )

        store = InMemoryVectorStore.from_documents(
            docs,
            EMBEDDINGS
        )

        relevant_docs = store.similarity_search(
            message,
            k=3
        )

        context = "\n\n".join(
            [d.page_content for d in relevant_docs]
        )

        cur.execute(
            """
            INSERT INTO chat_history
            (session_id,user_id,role,content)
            VALUES(%s,%s,%s,%s)
            """,
            (
                session_id,
                user_id,
                "user",
                message
            )
        )

        cur.execute(
            """
            SELECT role,content
            FROM chat_history
            WHERE session_id=%s
            ORDER BY created_at ASC
            """,
            (session_id,)
        )

        history = cur.fetchall()

        messages = []

        for item in history:

            if item["role"] == "user":

                messages.append(
                    HumanMessage(
                        content=item["content"]
                    )
                )

            else:

                messages.append(
                    AIMessage(
                        content=item["content"]
                    )
                )

        messages.append(
            HumanMessage(
                content=f"""
You are an AI assistant.

Use the PDF context below.

PDF Context:
{context}

Question:
{message}
"""
            )
        )

        response = MODEL.invoke(
            messages
        )

        answer = response.content

        cur.execute(
            """
            INSERT INTO chat_history
            (session_id,user_id,role,content)
            VALUES(%s,%s,%s,%s)
            """,
            (
                session_id,
                user_id,
                "assistant",
                answer
            )
        )

        db.commit()

        return {
            "response": answer
        }

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ==================================================
# LOAD CHAT HISTORY
# ==================================================

@app.get("/history/{session_id}")
async def history(
    session_id: str,
    db=Depends(get_db)
):

    cur = db.cursor()

    cur.execute(
        """
        SELECT role,content
        FROM chat_history
        WHERE session_id=%s
        ORDER BY created_at ASC
        """,
        (session_id,)
    )

    rows = cur.fetchall()

    return rows

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )