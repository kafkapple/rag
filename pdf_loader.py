import os
from typing import List, Tuple, Dict
import fitz  # PyMuPDF
import logging
from functools import lru_cache
import json
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """특정 폴더에서 PDF 문서를 로드하고 텍스트를 추출하는 클래스"""
    
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self.texts_folder = os.path.join(os.path.dirname(pdf_folder), "extracted_texts")
        os.makedirs(self.texts_folder, exist_ok=True)
        self._cached_files = set()

    def _save_extracted_text(self, title: str, content: str) -> None:
        """추출된 텍스트를 JSON 파일로 저장"""
        output_path = os.path.join(self.texts_folder, f"{title}.json")
        data = {
            "title": title,
            "content": content,
            "extracted_at": str(datetime.now()),
            "source_pdf": f"{title}.pdf"
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"텍스트 저장됨: {output_path}")

    def get_saved_text(self, title: str) -> Dict:
        """저장된 텍스트 불러오기"""
        file_path = os.path.join(self.texts_folder, f"{title}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    @lru_cache(maxsize=None)
    def _load_single_pdf(self, filepath: str) -> str:
        """단일 PDF 파일 로드 (캐시 적용)"""
        logger.info(f"PDF 파일 로딩 중: {os.path.basename(filepath)}")
        try:
            doc = fitz.open(filepath)
            content = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text.strip():  # 빈 페이지 제외
                        content.append(text)
                except Exception as e:
                    logger.error(f"페이지 {page_num+1} 텍스트 추출 실패 ({os.path.basename(filepath)}): {str(e)}")
            
            doc.close()
            return "\n\n".join(content)
            
        except Exception as e:
            logger.error(f"PDF 파일 로드 실패 ({os.path.basename(filepath)}): {str(e)}")
            return ""

    def load_pdfs(self) -> List[Tuple[str, str]]:
        """PDF 파일들을 로드하여 (제목, 내용) 튜플의 리스트로 반환"""
        if not os.path.exists(self.pdf_folder):
            logger.warning(f"PDF 폴더가 존재하지 않습니다: {self.pdf_folder}")
            return []

        current_files = set(f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf'))
        documents = []

        # 새로운 파일만 처리
        new_files = current_files - self._cached_files
        if not new_files:
            logger.info("새로운 PDF 파일이 없습니다.")
            return documents

        for filename in new_files:
            filepath = os.path.join(self.pdf_folder, filename)
            title = os.path.splitext(filename)[0]
            
            content = self._load_single_pdf(filepath)
            if content:
                documents.append((title, content))
                self._cached_files.add(filename)
                # 추출된 텍스트 저장
                self._save_extracted_text(title, content)
                logger.info(f"성공적으로 로드됨: {filename}")
            else:
                logger.warning(f"텍스트를 추출할 수 없음: {filename}")

        logger.info(f"총 {len(documents)}개의 새로운 PDF 문서가 로드되었습니다.")
        return documents
