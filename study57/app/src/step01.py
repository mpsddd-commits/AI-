from settings import settings
# LangChain 라이브러리에서 Ollama 모델과 통신하기 위한 클래스를 불러옵니다.
from langchain_ollama import OllamaLLM

def run(model: str = "gemma3:4b ", question: str = "한국에서 가장 높은 산은?"):
  """
  Ollama를 통해 모델에 질문을 던지고 답변을 출력하는 함수입니다.
  :param model: 사용할 모델의 명칭 (기본값: llama3.2 30억 파라미터 버전)
  :param question: AI에게 물어볼 질문 내용
  """
  
  # 1. OllamaLLM 객체를 생성합니다.
  # model: 사용할 모델의 이름을 지정합니다.
  # base_url: Ollama 서버가 실행 중인 주소를 전달합니다 (예: http://localhost:11434).
  llm = OllamaLLM(
    model=model, 
    base_url=settings.ollama_base_url
  )
  
  # 2. invoke 메서드를 사용하여 질문(Prompt)을 모델에 전달하고 답변을 받습니다.
  # 이 과정에서 로컬에 실행 중인 Ollama 엔진이 추론을 수행합니다.
  response = llm.invoke(question)
  
  # 3. 모델이 생성한 답변을 콘솔창에 출력합니다.
  print(response)

if __name__ == "__main__":
  run()
