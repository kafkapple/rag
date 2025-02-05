import streamlit as st
from pdf_loader import PDFLoader
from vector_store import VectorStore
from rag_model import RAGModel

# 설정
PDF_FOLDER = "data/"       # PDF 문서 폴더
FAISS_DB_PATH = "faiss_db/"  # FAISS 저장 경로
LMSTUDIO_API_URL = "http://172.31.249.207:1234/v1"#"http://localhost:1234/v1"  # LM Studio API URL

# PDF 문서 로딩
pdf_loader = PDFLoader(PDF_FOLDER)
documents = pdf_loader.load_pdfs()

# FAISS 벡터 DB 설정
vector_store = VectorStore(FAISS_DB_PATH)
vector_store.create_vector_store(documents)  # 최초 실행 시 필요
vector_store.load_vector_store()

# LM Studio 모델 설정
rag_model = RAGModel(LMSTUDIO_API_URL)

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
