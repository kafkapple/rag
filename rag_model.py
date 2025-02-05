from typing import List, Optional, Literal
import requests
from llama_cpp import Llama
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import os

logger = logging.getLogger(__name__)

class RAGModel:
    """LM Studio API 또는 로컬 모델을 사용하는 RAG 모델 클래스"""

    def __init__(
        self, 
        model_type: Literal["api", "local"],
        api_url: Optional[str] = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        top_p: float = 0.9
    ):
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.top_p = top_p
        
        if model_type == "api":
            if not api_url:
                raise ValueError("API URL must be provided when using API model")
            self.api_url = api_url
            self.model_name = model_name
        else:
            if not model_path:
                raise ValueError("model_path must be provided when using local model")
            
            # 모델 경로 처리
            if isinstance(model_path, str) and '/' in model_path:
                # Hugging Face 형식 경로인 경우
                try:
                    # config 참조 제거
                    repo_id, filename = model_path.rsplit('/', 1)
                    local_model_path = self._ensure_model_downloaded(repo_id, filename)
                except Exception as e:
                    logger.error(f"Failed to process model path: {str(e)}")
                    raise ValueError(f"Invalid model path format: {model_path}")
            else:
                # 로컬 경로인 경우
                local_model_path = model_path
                
            if not os.path.exists(local_model_path):
                raise ValueError(f"Model file not found at: {local_model_path}")
                
            logger.info(f"Loading model from: {local_model_path}")
            self.llm = Llama(
                model_path=local_model_path,
                n_threads=4
            )
            logger.info("Model loaded successfully")

    def _ensure_model_downloaded(self, repo_id: str, filename: str) -> str:
        """Hugging Face에서 모델을 다운로드하고 로컬 경로 반환"""
        # repo_id를 기반으로 모델 저장 경로 생성
        models_dir = Path("models")
        repo_dir = models_dir / repo_id.replace('/', '_')  # '/'를 '_'로 변환하여 폴더명으로 사용
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Checking model in: {repo_dir}")
            model_path = repo_dir / filename
            
            # 이미 다운로드된 모델이 있는지 확인
            if model_path.exists():
                logger.info(f"Model already exists at: {model_path}")
                return str(model_path)
            
            # 모델 다운로드
            logger.info(f"Downloading model from {repo_id}/{filename}")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=repo_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise ValueError(f"Failed to download model from {repo_id}/{filename}: {str(e)}")

    def _create_prompt(self, query: str, docs: List) -> str:
        """프롬프트 생성"""
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return f"""주어진 문맥을 바탕으로 질문에 답변해주세요. 문맥에서 답을 찾을 수 없다면, '주어진 문맥에서 답을 찾을 수 없습니다.'라고 답변해주세요.

문맥:
{context}

질문: {query}

답변: 주어진 문맥을 바탕으로,"""

    def generate_answer(self, query: str, docs: List) -> str:
        """문서를 참고하여 질문에 답변 생성"""
        prompt = self._create_prompt(query, docs)
        
        try:
            if self.model_type == "api":
                try:
                    logger.info(f"Sending request to API: {self.api_url}")
                    
                    # API 요청 파라미터 구성
                    request_data = {
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "repeat_penalty": self.repeat_penalty,
                        "top_p": self.top_p,
                        "stop": ["\n\n", "질문:", "문맥:"]
                    }
                    
                    # model_name이 있는 경우에만 추가
                    if hasattr(self, 'model_name') and self.model_name:
                        request_data["model"] = self.model_name
                    
                    logger.debug(f"API request parameters: {request_data}")
                    
                    response = requests.post(
                        f"{self.api_url}/completions",
                        json=request_data
                    )
                    
                    # API 응답 로깅
                    logger.info(f"API response status: {response.status_code}")
                    logger.debug(f"API raw response: {response.text}")
                    
                    if response.status_code != 200:
                        logger.error(f"API error: {response.text}")
                        return f"API 오류가 발생했습니다. (Status: {response.status_code})"
                    
                    response_json = response.json()
                    logger.info(f"API response structure: {response_json.keys()}")
                    
                    # API 응답 형식 검증
                    if "choices" not in response_json:
                        logger.error(f"Unexpected API response format: {response_json}")
                        return "API 응답 형식이 올바르지 않습니다."
                    
                    return response_json["choices"][0]["text"].strip()
                    
                except requests.exceptions.RequestException as req_err:
                    logger.error(f"API request failed: {str(req_err)}")
                    return "API 요청 중 오류가 발생했습니다."
                except ValueError as val_err:
                    logger.error(f"JSON parsing error: {str(val_err)}")
                    return "API 응답을 처리하는 중 오류가 발생했습니다."
            
            else:
                try:
                    # 모델 상태 확인
                    logger.info(f"Model path: {self.llm.model_path}")
                    logger.info(f"Model loaded: {hasattr(self.llm, '_model')}")
                    
                    # 입력값 로깅
                    logger.debug(f"Input prompt length: {len(prompt)}")
                    logger.debug(f"Parameters: max_tokens={self.max_tokens}, temp={self.temperature}")
                    
                    response = self.llm(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        repeat_penalty=self.repeat_penalty,
                        top_p=self.top_p,
                        stop=["\n\n", "질문:", "문맥:"],
                        echo=False
                    )
                    
                    # 상세 응답 로깅
                    logger.info(f"Raw response type: {type(response)}")
                    logger.info(f"Raw response: {response}")
                    
                    if isinstance(response, dict):
                        logger.info(f"Response keys: {response.keys()}")
                        if "choices" in response:
                            logger.info(f"Choices structure: {response['choices']}")
                        return self._extract_text_from_response(response)
                    else:
                        logger.warning(f"Unexpected response type: {type(response)}")
                        return str(response)
                    
                except Exception as local_err:
                    logger.error(f"Local model error details:", exc_info=True)
                    logger.error(f"Error type: {type(local_err)}")
                    logger.error(f"Error args: {local_err.args}")
                    raise
            
        except Exception as e:
            logger.error(f"Final error: {str(e)}", exc_info=True)
            return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

    def _extract_text_from_response(self, response: dict) -> str:
        """응답에서 텍스트 추출을 위한 헬퍼 함수"""
        logger.debug(f"Extracting text from response: {response}")
        
        if "choices" in response and response["choices"]:
            if isinstance(response["choices"][0], dict):
                return response["choices"][0].get("text", "").strip()
            return str(response["choices"][0]).strip()
        
        if "generated_text" in response:
            return response["generated_text"].strip()
        
        logger.warning(f"No known text field found in response: {response}")
        return str(response)
