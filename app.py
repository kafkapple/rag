import os
import streamlit as st
from pdf_loader import PDFLoader
from vector_store import VectorStore
from rag_model import RAGModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 설정
PDF_FOLDER = os.getenv("PDF_FOLDER", "data/")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "faiss_db/")
LMSTUDIO_API_URL = "http://localhost:1234/v1"
LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Create necessary directories
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# PDF 문서 로딩
pdf_loader = PDFLoader(PDF_FOLDER)
documents = pdf_loader.load_pdfs()

# FAISS 벡터 DB 설정
vector_store = VectorStore(FAISS_DB_PATH)
if documents:  # Only create if there are documents
    vector_store.create_vector_store(documents)
vector_store.load_vector_store()

# 모델 선택
model_option = st.sidebar.radio(
    "🤖 모델 선택",
    ["LM Studio API", "로컬 모델"]
)

if model_option == "LM Studio API":
    rag_model = RAGModel(model_type="api", api_url=LMSTUDIO_API_URL)
else:
    rag_model = RAGModel(model_type="local", model_path=LOCAL_MODEL_PATH)

# Streamlit UI
st.title("📚 RAG QA System - PDF 문서 기반 검색")

query = st.text_input("💡 질문을 입력하세요:")

if query:
    with st.spinner("🔎 문서 검색 중..."):
        retrieved_docs = vector_store.search(query, k=3)
        answer = rag_model.generate_answer(query, retrieved_docs)

    st.write("## 📝 답변:")
    st.write(answer)

    st.write("## 📄 참고 문서:")
    for doc in retrieved_docs:
        st.write(doc.page_content[:500] + "...")
