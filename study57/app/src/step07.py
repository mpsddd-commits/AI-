from settings import settings
# List를 임포트하여 변수가 단순 문자열이 아닌 '문자열 담긴 리스트'임을 명시합니다.
# 이는 LLM이 도구를 사용할 때 인자값의 형식을 정확히 파악하도록 돕습니다.
from typing import List
# AIMessage는 LLM(인공지능)이 생성한 응답을 담는 클래스입니다.
# 일반 텍스트뿐만 아니라 모델이 결정한 '도구 호출(tool_calls)' 정보도 이 객체에 담깁니다.
from langchain.messages import AIMessage
# @tool 데코레이터를 일반 파이썬 함수 위에 붙이면, 
# LangChain이 함수의 이름, 설명(Docstring), 인자 타입을 추출해 AI에게 전달할 수 있는 형태로 바꿉니다.
from langchain.tools import tool
# ChatOllama는 '대화형' 모델에 특화된 클래스입니다.
# 시스템 메시지 설정, 도구 바인딩(bind_tools) 등 고급 기능을 지원합니다.
from langchain_ollama import ChatOllama

# 1. 도구(Tool) 정의
# @tool 데코레이터를 사용하여 일반 파이썬 함수를 LangChain이 이해할 수 있는 '도구'로 변환합니다.
# 함수의 문서화 문자열(Docstring)은 LLM이 이 도구를 언제, 어떻게 사용할지 판단하는 기준이 됩니다.
@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
  """과거 주소 목록을 사용하여 사용자를 검증합니다.

  Args:
    user_id (int): 사용자의 고유 ID.
    addresses (List[str]): 문자열 리스트 형식의 이전 주소들.
  """
  # 실제 환경에서는 여기서 DB 조회 등을 수행하겠지만, 여기서는 예시를 위해 True를 반환합니다.
  return True

def run(model: str = "llama3.2:3b"):
  # 2. ChatOllama 모델 설정
  # validate_model_on_init: 모델이 로드되었는지 초기화 시 확인합니다.
  # temperature=0: 모델의 창의성을 낮추고 일관된(결정론적인) 답변을 하도록 설정합니다 (도구 호출 시 권장).
  # .bind_tools([validate_user]): 위에서 정의한 함수를 모델에 연결합니다. 
  # 이제 모델은 질문을 듣고 이 함수를 호출할지 스스로 결정할 수 있습니다.
  llm = ChatOllama(
    model=model,
    base_url=settings.ollama_base_url,
    validate_model_on_init=True,
    temperature=0,
  ).bind_tools([validate_user])

  # 3. 질문 수행
  # 사용자가 자연어로 질문을 던집니다. (ID 123인 사용자와 주소 정보 포함)
  result = llm.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
  )

  # 4. 결과 분석 (Tool Call 확인)
  # 모델의 응답(result)이 AIMessage 객체이고, 도구 호출(tool_calls) 정보가 포함되어 있는지 확인합니다.
  if isinstance(result, AIMessage) and result.tool_calls:
    # 모델은 "알겠습니다"라는 텍스트 대신, 
    # 'validate_user' 함수를 'user_id=123', 'addresses=[...]' 인자로 호출하라는 구조화된 데이터를 반환합니다.
    print("모델의 도구 호출 결정:")
    print(result.tool_calls)

if __name__ == "__main__":
  run()