from dataclasses import dataclass
from typing import List
from omegaconf import DictConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch

@dataclass
class Document:
    """문서 정보를 담는 클래스"""
    title: str
    content: str
    chunks: List[str]

class VectorStore:
    """FAISS 벡터 데이터베이스를 관리하는 클래스"""

    def __init__(self, db_path: str, cfg: DictConfig = None):
        self.db_path = db_path
        
        # CUDA가 사용 가능한 경우 GPU 사용
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=cfg.vector_store.model_name if cfg else "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        self.vector_store = None
        
        chunk_size = cfg.vector_store.chunk_size if cfg else 500
        chunk_overlap = cfg.vector_store.chunk_overlap if cfg else 50
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.documents = {}  # title을 키로 하는 Document 객체 저장

    def create_vector_store(self, documents: List[tuple]):
        """문서 임베딩 후 FAISS 벡터 저장
        
        Args:
            documents: (title, content) 튜플의 리스트
        """
        all_chunks = []
        print("문서를 청크로 분할 중...")
        
        for title, content in tqdm(documents, desc="문서 분할"):
            chunks = self.text_splitter.split_text(content)
            all_chunks.extend(chunks)
            
            # Document 객체 생성 및 저장
            self.documents[title] = Document(
                title=title,
                content=content,
                chunks=chunks
            )
        
        print("벡터 저장소 생성 중...")
        self.vector_store = FAISS.from_texts(
            tqdm(all_chunks, desc="텍스트 임베딩"),
            self.embedding_model
        )
        self.vector_store.save_local(self.db_path)
        print("벡터 저장소가 성공적으로 저장되었습니다.")

    def load_vector_store(self):
        """저장된 FAISS 벡터 DB 로드"""
        print("FAISS 벡터 저장소 로딩 중...")
        self.vector_store = FAISS.load_local(
            self.db_path, 
            self.embedding_model,
            allow_dangerous_deserialization=True  # 로컬 환경에서 신뢰할 수 있는 파일이므로 허용
        )
        print("벡터 저장소가 성공적으로 로드되었습니다.")

    def search(self, query, k=3):
        """FAISS를 이용해 검색된 문서 반환"""
        if not self.vector_store:
            raise ValueError("FAISS 벡터 스토어가 로드되지 않았습니다.")
        
        # Convert search results to the expected format
        docs = self.vector_store.similarity_search(query, k)
        return [type('Document', (), {'page_content': doc.page_content}) for doc in docs]
