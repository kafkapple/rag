from langchain.llms import LlamaCpp

class RAGModel:
    """LM Studio API를 통해 LLaMA, Qwen, DeepSeek 모델을 호출하는 클래스"""

    def __init__(self, api_url: str):
        self.llm = LlamaCpp(
            model_path=api_url,
            temperature=0.7,
            max_tokens=512
        )

    def generate_answer(self, query, retrieved_docs):
        """검색된 문서를 컨텍스트로 활용하여 답변 생성"""
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        return self.llm(prompt)
