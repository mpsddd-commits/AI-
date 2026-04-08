from settings import settings
# LangChain에서 채팅 전용 Ollama 인터페이스를 불러옵니다.
# (OllamaLLM과 달리 메시지 객체 구조를 처리하는 데 최적화되어 있습니다.)
from langchain_ollama import ChatOllama

def run(model: str = "llama3.2:3b", question: str = "한국에서 가장 높은 산은?"):
  """
  채팅 모델을 사용하여 시스템 메시지(역할 설정)와 사용자 질문을 처리하는 함수입니다.
  """
  
  # 1. ChatOllama 인스턴스 생성
  # 채팅 인터페이스를 통해 모델과 통신하도록 설정합니다.
  llm = ChatOllama(
    model=model, 
    base_url=settings.ollama_base_url
  )
  
  # 2. 대화 메시지 구성 (Message List)
  # 각 튜플은 (역할, 내용)으로 구성됩니다. => 프롬프트를 두가지로 나눈다
  # - system: AI의 성격이나 지침을 설정 (여기서는 번역기 역할)
  # - human: 실제 사용자가 입력하는 질문 내용
  messages = [
    (
      "system", 
      "You are a useful helper that translates Korean into English. "
      "Please translate the sentence entered by the user."
    ),
    ("human", question),
  ]
  
  # 3. 모델 호출 (Invoke)
  # 단순 문자열이 아닌 '메시지 리스트'를 전달하여 맥락이 포함된 응답을 받습니다.
  response = llm.invoke(messages)
  
  # 4. 결과 출력
  # response는 단순 문자열이 아니라 메타데이터가 포함된 AIMessage 객체입니다.
  print("응답 객체 전체:", response)
  
  # response.content 속성을 통해 실제 AI가 생성한 텍스트 내용만 추출합니다.
  print("응답 내용:", response.content)

if __name__ == "__main__":
  run()