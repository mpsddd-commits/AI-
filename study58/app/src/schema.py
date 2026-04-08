# pydantic 라이브러리에서 BaseModel을 불러옵니다.
# Pydantic은 실행 시점에 데이터 타입을 강제하고 검증(Validation)하는 역할을 합니다.
from pydantic import BaseModel
# typing 모듈에서 TypedDict를 불러옵니다.
# TypedDict는 딕셔너리에 들어갈 '키(Key)'와 '값(Value)의 타입'을 명확히 정의할 때 사용합니다.
from typing import TypedDict


# 1. 데이터 구조 정의
# User라는 이름의 새로운 타입을 정의합니다. 
# 이 구조를 따르는 딕셔너리는 반드시 아래의 세 가지 키를 가져야 하며, 정해진 타입을 준수해야 합니다.
class User1(TypedDict):
  id: int      # 사용자 ID (정수형)
  name: str    # 사용자 이름 (문자열)
  email: str   # 사용자 이메일 (문자열)
  
# 2. Pydantic 모델 정의
# 클래스 형식으로 정의하며, 각 필드의 타입 힌트를 지정합니다.
# 이 모델은 나중에 딕셔너리 데이터를 받아 유효성을 검사하는 '틀'이 됩니다.
class User2(BaseModel):
  id: int      # 사용자 ID (정수형)
  name: str    # 사용자 이름 (문자열)
  email: str   # 사용자 이메일 (문자열)
