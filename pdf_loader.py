import os
import fitz  # PyMuPDF

class PDFLoader:
    """특정 폴더에서 PDF 문서를 로드하고 텍스트를 추출하는 클래스"""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트를 추출"""
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text

    def load_pdfs(self):
        """폴더 내 모든 PDF 파일을 로드하고 텍스트 추출"""
        documents = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)
                text = self.extract_text_from_pdf(file_path)
                documents.append(text)
        return documents
