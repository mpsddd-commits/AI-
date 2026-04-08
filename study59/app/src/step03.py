from pydantic import BaseModel, Field
from settings import settings
from langchain_ollama import ChatOllama

# 1. 모델 설정
model_name: str = "gemma4:e4b"

# ChatOllama: Ollama를 통해 실행되는 챗 모델 인터페이스
llm = ChatOllama(
  model=model_name, 
  base_url=settings.ollama_base_url,
  # format="json": 모델이 유효한 JSON 형식으로만 답변하도록 강제합니다.
  # structured_output 사용 시 파싱 오류를 줄이는 데 매우 중요합니다.
  format="json",
  # temperature=0: 모델의 창의성을 최소화하고 일관된 답변을 생성하게 합니다. (정보 추출 시 권장)
  temperature=0
)

# 2. 출력 스키마 정의: LLM이 어떤 형태의 데이터를 반환해야 하는지 정의
class MovieResponse(BaseModel):
  """사용자에게 영화 정보를 제공할 때 사용하는 데이터 구조입니다."""

  # Field를 사용하여 각 필드의 의미를 설명합니다. 
  # LLM은 이 설명을 보고 어떤 값을 추출할지 판단합니다.
  title: str = Field(description="영화 제목")
  director: str = Field(description="감독 이름")
  genre: str = Field(description="장르")
  release_year: int = Field(description="개봉 연도")

def run():
  try:
    # 3. 구조화된 출력기 생성
    # with_structured_output: 입력받은 Pydantic 클래스(MovieResponse)에 맞춰  !!chatollama랑 format="json"이랑 같이 써야함!! 그래야 .with_structured_output 사용가능
    # LLM의 답변을 자동으로 파싱하여 객체로 변환해주는 기능을 활성화합니다.
    structured_llm = llm.with_structured_output(MovieResponse)
    
    # 4. 모델 호출 및 처리
    # 타이타닉에 대한 긴 설명을 요청하더라도, 모델은 MovieResponse 형식에 맞는 데이터만 추출합니다.
    response = structured_llm.invoke("click 영화에 대해 설명해주세요.")
    
    # 5. 결과 출력
    # response는 더 이상 단순 문자열이 아니라 MovieResponse 인스턴스(객체)입니다.
    # 따라서 .title, .director 처럼 속성값으로 바로 접근이 가능합니다.
    print("\n--- 추출된 영화 정보 ---")
    print(f"제목: {response.title}")
    print(f"감독: {response.director}")
    print(f"장르: {response.genre}")
    print(f"개봉 연도: {response.release_year}")
      
  except Exception as e:
    # 모델이 JSON 형식을 지키지 못했거나, 필수 필드를 누락했을 경우 예외가 발생할 수 있습니다.
    print(f"오류 발생: {e}")

if __name__ == '__main__':
  run()
