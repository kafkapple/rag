import os
import streamlit as st
from pdf_loader import PDFLoader
from vector_store import VectorStore
from rag_model import RAGModel
from dotenv import load_dotenv
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

def get_config():
    """Hydra 설정을 로드하는 함수"""
    try:
        GlobalHydra.instance().clear()
        with hydra.initialize(version_base=None, config_path="config"):
            cfg = hydra.compose(config_name="config")
        return cfg
    except:
        # 에러 발생 시 기본 설정 반환
        return OmegaConf.create({
            "model": {
                "lm_studio": {
                    "api_url": "http://172.31.249.207:1234/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "repeat_penalty": 1.1,
                    "top_p": 0.9
                },
                "local": {
                    "path": "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
                    "max_tokens": 8192,
                    "temperature": 0.7,
                    "repeat_penalty": 1.1,
                    "top_p": 0.9
                }
            },
            "vector_store": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "paths": {
                "pdf_folder": "data/"
            }
        })

def main():
    # Load environment variables
    load_dotenv()
    
    # Hydra 설정을 세션 상태로 관리
    if 'config' not in st.session_state:
        st.session_state.config = get_config()
    
    cfg = st.session_state.config

    # Create necessary directories
    os.makedirs(cfg.paths.pdf_folder, exist_ok=True)
    
    # PDF 문서와 벡터 스토어를 세션 상태로 관리
    if 'documents' not in st.session_state:
        st.write("PDF 문서 로딩 중...")
        pdf_loader = PDFLoader(cfg.paths.pdf_folder)
        st.session_state.documents = pdf_loader.load_pdfs()
        st.session_state.pdf_loader = pdf_loader  # PDF 로더 인스턴스 저장
    
    if 'vector_store' not in st.session_state:
        vector_store = VectorStore("faiss_db", cfg)  # 직접 경로 지정
        if st.session_state.documents:  # Only create if there are documents
            with st.spinner("벡터 저장소 생성 중..."):
                vector_store.create_vector_store(st.session_state.documents)
        with st.spinner("벡터 저장소 로딩 중..."):
            vector_store.load_vector_store()
        st.session_state.vector_store = vector_store
    
    vector_store = st.session_state.vector_store

    # 모델 선택
    model_option = st.sidebar.radio(
        "🤖 모델 선택",
        ["LM Studio API", "로컬 모델"]
    )

    if model_option == "LM Studio API":
        rag_model = RAGModel(
            model_type="api", 
            api_url=cfg.model.lm_studio.api_url
        )
    else:
        rag_model = RAGModel(
            model_type="local", 
            model_path=cfg.model.local.path
        )

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
        
        # 검색된 청크들을 문서별로 그룹화
        grouped_docs = {}
        for doc in retrieved_docs:
            # 문서 제목 찾기
            for title, document in vector_store.documents.items():
                if doc.page_content in document.chunks:
                    if title not in grouped_docs:
                        grouped_docs[title] = []
                    grouped_docs[title].append(doc.page_content)

        # 문서별로 표시
        for title, chunks in grouped_docs.items():
            expander = st.expander(f"📑 {title}")
            with expander:
                st.write("### 문서 개요")
                st.write(vector_store.documents[title].content[:300] + "...")
                
                st.write("### 관련 구절")
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**구절 {i}:**")
                    st.write(chunk)
                    st.divider()

    # 사이드바에 원본 텍스트 보기 옵션 추가
    with st.sidebar:
        st.write("## 📑 원본 문서 보기")
        for title, _ in st.session_state.documents:
            if st.button(f"📄 {title}"):
                saved_text = st.session_state.pdf_loader.get_saved_text(title)
                if saved_text:
                    with st.expander(f"원본 텍스트 - {title}", expanded=True):
                        st.text(f"추출 시간: {saved_text['extracted_at']}")
                        st.text(f"원본 파일: {saved_text['source_pdf']}")
                        st.markdown("---")
                        st.markdown(saved_text['content'])

if __name__ == "__main__":
    main()
