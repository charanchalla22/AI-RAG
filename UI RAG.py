import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import tempfile
load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",
    reasoning_format="parsed"
)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None
MAX_HISTORY_LEN = 6

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)
if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.vector_store = None
        st.session_state.chat_history = []

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Loading PDF and creating embeddings..."):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        total_chars = sum(
            len(doc.page_content.strip())
            for doc in docs
        )
        if total_chars < 50:
            st.error(
                "No meaningful text found in the PDF. It may be a scanned/image PDF."
            )
            st.stop()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = InMemoryVectorStore.from_documents(
            all_splits,
            embeddings
        )
        st.session_state.vector_store = vector_store
    st.success("PDF processed successfully!")
def retrieve_context(query: str) -> str:
    similar_docs = st.session_state.vector_store.similarity_search(
        query,
        k=3
    )
    data = []
    for doc in similar_docs:
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")
        data.append(
            f"Content: {content}\nSource: {source}"
        )
    return "\n\n".join(data)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question using only the provided context."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "<context>\n{context}\n</context>\n\nQuestion: {question}"
    )
])
chain = prompt | model | StrOutputParser()
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
if st.session_state.vector_store is not None:
    query = st.chat_input(
        "Ask a question about the PDF"
    )
    if query:
        with st.chat_message("user"):
            st.write(query)
        context = retrieve_context(query)
        active_history = st.session_state.chat_history[
            -MAX_HISTORY_LEN:
        ]
        response = chain.invoke({
            "context": context,
            "chat_history": active_history,
            "question": query
        })
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.chat_history.append(
            HumanMessage(content=query)
        )
        st.session_state.chat_history.append(
            AIMessage(content=response)
        )