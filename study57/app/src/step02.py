from settings import settings
# LangChain 라이브러리에서 Ollama와 연동하기 위한 클래스를 불러옵니다.
from langchain_ollama import OllamaLLM

def run(model: str = "llama3.2:3b", question: str = "한국에서 가장 높은 산은?"):
  """
  Ollama 모델을 호출하여 질문에 대한 답변을 실시간 스트리밍으로 출력하는 함수입니다.
  """
  
  # 1. OllamaLLM 인스턴스 생성
  # model: 사용할 언어 모델 이름
  # base_url: Ollama 서버 주소 (settings 파일에 정의된 값 사용)
  llm = OllamaLLM(
    model=model, 
    base_url=settings.ollama_base_url
  )
  
  # 2. 사용자에게 답변 시작을 알리는 메시지 출력
  # end="": 줄바꿈을 하지 않고 바로 뒤에 답변을 붙여넣기 위함
  # flush=True: 버퍼에 쌓아두지 않고 즉시 화면에 출력
  print(f"답변 중 ({model}): ", end="", flush=True)
  
  # 3. llm.stream(question)을 사용하여 답변을 조각(chunk) 단위로 받아옵니다.
  # 모델이 답변을 생성하는 즉시 한 글자 혹은 한 단어씩 반환됩니다.
  for chunk in llm.stream(question):
    # 받아온 답변 조각(chunk)을 화면에 연속해서 출력합니다.
    # end="": 조각들 사이에 줄바꿈이 생기지 않도록 설정
    # flush=True: 실시간으로 글자가 타이핑되는 듯한 효과를 위해 출력 스트림을 즉시 비움
    print(chunk, end="", flush=True)
  
  # 4. 모든 답변이 끝나면 마지막에 줄바꿈을 한 번 해줍니다.
  print()

if __name__ == "__main__":
  run()