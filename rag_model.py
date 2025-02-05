from langchain_openai import ChatOpenAI
import time

class RAGModel:
    """LM Studio API를 통해 LLaMA, Qwen, DeepSeek 모델을 호출하는 클래스"""

    def __init__(self, api_url: str):
        print("🔄 LLM 모델 초기화 중...")
        self.llm = ChatOpenAI(
            base_url=api_url,
            api_key="not-needed",
            temperature=0.7,
            max_tokens=512,
            request_timeout=30  # 타임아웃 설정 추가
        )
        print("✅ LLM 모델 초기화 완료")

    def generate_answer(self, query, retrieved_docs):
        """검색된 문서를 컨텍스트로 활용하여 답변 생성"""
        print("🤖 답변 생성 중...")
        # 컨텍스트 길이 제한
        max_context_length = 3000  # 적절한 값으로 조정
        context = "\n".join([doc.page_content[:max_context_length] for doc in retrieved_docs])
        
        prompt = (
            "다음 문서 내용을 바탕으로 질문에 답변해주세요.\n\n"
            f"문서 내용: {context}\n\n"
            f"질문: {query}\n\n"
            "답변:"
        )
        
        try:
            start_time = time.time()
            response = self.llm.invoke(prompt)
            end_time = time.time()
            print(f"✅ 답변 생성 완료 (소요시간: {end_time - start_time:.2f}초)")
            return response.content
        except Exception as e:
            print(f"❌ 답변 생성 중 오류 발생: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
