from settings import settings
# Ollama 모델과의 통신을 위한 클래스를 불러옵니다.
from langchain_ollama import OllamaLLM
# 프롬프트의 구조(틀)를 정의하기 위한 클래스를 불러옵니다.
from langchain_core.prompts import PromptTemplate

def run(model: str = "llama3.2:3b", question: str = "한국에서 가장 높은 산은?"):
  """
  프롬프트 템플릿과 LLM을 체인으로 엮어 실행하는 함수입니다.
  """
  
  # 1. Ollama LLM 객체 생성
  # 로컬 서버 주소와 사용할 모델명을 설정합니다.
  llm = OllamaLLM(
    model=model, 
    base_url=settings.ollama_base_url
  )
  
  # 2. 프롬프트 템플릿(PromptTemplate) 정의
  # {question} 부분은 나중에 실제 질문 내용으로 치환될 변수입니다.
  # 이를 통해 사용자 입력을 특정 형식(예: "다음 질문에 답하세요: ...") 안에 넣을 수 있습니다.
  prompt = PromptTemplate.from_template("다음 질문에 답하세요: {question}")
  
  # 3. LangChain Expression Language (LCEL)를 이용한 체인 구성
  # '|' (파이프) 연산자를 사용하여 프롬프트와 LLM을 하나로 묶습니다.
  # prompt의 결과물이 llm의 입력으로 자동으로 전달됩니다.
  chain = prompt | llm 
  
  # 4. 체인 실행 (Invoke)
  # 템플릿에 정의된 변수명인 'question'을 키(Key)로 하여 딕셔너리 형태로 값을 전달합니다.
  # 내부 흐름: 질문 -> 프롬프트 완성 -> LLM 전달 -> 답변 생성
  response = chain.invoke({"question": question})
  
  # 5. 최종 답변 출력
  print(response)

if __name__ == "__main__":
  run()