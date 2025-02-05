from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    """FAISS 벡터 데이터베이스를 관리하는 클래스"""

    def __init__(self, db_path: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def create_vector_store(self, documents):
        """문서 임베딩 후 FAISS 벡터 저장"""
        # Split documents into chunks
        chunks = []
        for doc in documents:
            doc_chunks = self.text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
        
        self.vector_store = FAISS.from_texts(chunks, self.embedding_model)
        self.vector_store.save_local(self.db_path)

    def load_vector_store(self):
        """저장된 FAISS 벡터 DB 로드"""
        self.vector_store = FAISS.load_local(self.db_path, self.embedding_model)

    def search(self, query, k=3):
        """FAISS를 이용해 검색된 문서 반환"""
        if not self.vector_store:
            raise ValueError("FAISS 벡터 스토어가 로드되지 않았습니다.")
        
        # Convert search results to the expected format
        docs = self.vector_store.similarity_search(query, k)
        return [type('Document', (), {'page_content': doc.page_content}) for doc in docs]
