from typing import Literal, Optional
import os
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

class RAGModel:
    """LM Studio API 또는 로컬 모델을 사용하는 RAG 모델 클래스"""

    def __init__(
        self, 
        model_type: Literal["api", "local"],
        api_url: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_window: int = 2048
    ):
        # Load environment variables
        load_dotenv()
        
        self.model_type = model_type
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if model_type == "api":
            if not api_url:
                raise ValueError("API URL must be provided when using API model")
        else:
            if not model_path:
                model_path = os.getenv("MODEL_PATH")
            
            # Create models directory if it doesn't exist
            Path("models").mkdir(exist_ok=True)
            
            # Download model if it doesn't exist
            if not os.path.exists(model_path):
                print(f"Downloading model to {model_path}...")
                hf_hub_download(
                    repo_id=os.getenv("HF_MODEL_ID"),
                    filename=Path(model_path).name,
                    local_dir="models",
                    token=os.getenv("HF_TOKEN")
                )
            
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                n_ctx=context_window,
                n_batch=512,
                verbose=True
            )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""아래 주어진 문맥을 바탕으로 질문에 답변해주세요.

문맥:
{context}

질문: {question}

답변:"""
        )

    def generate_answer(self, query, retrieved_docs):
        """검색된 문서를 컨텍스트로 활용하여 답변 생성"""
        # Limit context length to avoid exceeding context window
        max_context_length = 1024  # Leave room for prompt and response
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = self.prompt_template.format(context=context, question=query)
        
        if self.model_type == "api":
            response = requests.post(
                f"{self.api_url}/completions",
                json={
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            )
            return response.json()["choices"][0]["text"]
        else:
            return self.llm(prompt)
