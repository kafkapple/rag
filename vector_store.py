from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class VectorStore:
    """FAISS 벡터 데이터베이스를 관리하는 클래스"""

    def __init__(self, db_path: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None

    def create_vector_store(self, documents):
        """문서 임베딩 후 FAISS 벡터 저장"""
        self.vector_store = FAISS.from_texts(documents, self.embedding_model)
        self.vector_store.save_local(self.db_path)

    def load_vector_store(self):
        """저장된 FAISS 벡터 DB 로드"""
        self.vector_store = FAISS.load_local(self.db_path, self.embedding_model)

    def search(self, query, k=3):
        """FAISS를 이용해 검색된 문서 반환"""
        if not self.vector_store:
            raise ValueError("FAISS 벡터 스토어가 로드되지 않았습니다.")
        return self.vector_store.similarity_search(query, k)
