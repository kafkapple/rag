import os
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFLoader:
    """특정 폴더에서 PDF 문서를 로드하고 텍스트를 추출하는 클래스"""
    
    def __init__(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.folder_path = folder_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트를 추출"""
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text

    def load_pdfs(self):
        """폴더 내 모든 PDF 파일을 로드하고 텍스트 추출"""
        documents = []
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
        
        print("📚 PDF 문서 로딩 및 청크 분할 중...")
        for filename in tqdm(pdf_files, desc="PDF 처리 중"):
            file_path = os.path.join(self.folder_path, filename)
            text = self.extract_text_from_pdf(file_path)
            # 텍스트를 청크로 분할
            chunks = self.text_splitter.split_text(text)
            documents.extend(chunks)
        
        print(f"✅ {len(pdf_files)}개의 PDF에서 {len(documents)}개의 청크 생성 완료")
        return documents
