from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

class VectorStore:
    """FAISS 벡터 데이터베이스를 관리하는 클래스"""

    def __init__(self, db_path: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = db_path
        print("🔄 임베딩 모델 로딩 중...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        print("✅ 임베딩 모델 로딩 완료")
        self.vector_store = None

    def create_vector_store(self, documents):
        """문서 임베딩 후 FAISS 벡터 저장"""
        print("🔄 문서 임베딩 및 벡터 저장 중...")
        texts = []
        for doc in tqdm(documents, desc="문서 처리 중"):
            texts.append(doc)
        self.vector_store = FAISS.from_texts(texts, self.embedding_model)
        self.vector_store.save_local(self.db_path)
        print("✅ 벡터 저장소 생성 완료")

    def load_vector_store(self):
        """저장된 FAISS 벡터 DB 로드"""
        self.vector_store = FAISS.load_local(
            self.db_path, 
            self.embedding_model,
            allow_dangerous_deserialization=True  # 신뢰할 수 있는 로컬 데이터인 경우에만 True로 설정
        )

    def search(self, query, k=3):
        """FAISS를 이용해 검색된 문서 반환"""
        if not self.vector_store:
            raise ValueError("FAISS 벡터 스토어가 로드되지 않았습니다.")
        print("🔍 유사 문서 검색 중...")
        results = self.vector_store.similarity_search(query, k)
        print(f"✅ {len(results)}개의 관련 문서 검색 완료")
        return results
