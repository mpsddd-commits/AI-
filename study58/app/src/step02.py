from src.schema import User2 as User

def run():
  try:
    # 2. 테스트용 원본 데이터 (딕셔너리)
    # 주의: 'name' 값이 문자열이 아닌 정수(1234)로 되어 있습니다.
    user_data = {
      'id': 1,
      'name': "1234",  # 데이터 타입 불일치 발생 시도
      'email': 'test@example.com'
    }
    
    # 3. 모델 인스턴스 생성 (데이터 검증 시작)
    # **user_data는 딕셔너리를 '키=값' 형태로 풀어헤쳐서 전달합니다.
    # Pydantic의 특징: 'name'이 정수(1234)라도 문자열("1234")로 변환이 가능하면 
    # 자동으로 형변환(Coercion)을 시도하여 성공시킵니다.
    user1 = User(**user_data)
    
    # 성공적으로 검증된 객체를 출력합니다.
    print(f"검증 성공 결과: {user1}")
  except Exception as e:
    # 데이터 타입이 도저히 변환 불가능하거나(예: 문자열 자리에 리스트가 옴),
    # 필수 키가 빠진 경우 에러를 발생시킵니다.
    print(f"검증 에러 발생: {e}")

if __name__ == "__main__":
  run()
