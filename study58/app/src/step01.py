from src.schema import User1 as User

def run():
  try:
    # 2. TypedDict 인스턴스 생성
    # 변수명 뒤에 ': User'를 붙여 이 변수가 User 구조임을 명시합니다.
    # 만약 여기서 'email' 키를 빼먹거나, 'id'에 문자열을 넣으면 정적 분석 도구(Pyright, MyPy 등)가 경고를 줍니다.
    user1: User = {
      'id': 1,
      'name': 'test',
      'email': 'test@example.com'
    }
    
    # 생성된 딕셔너리 객체를 출력합니다.
    print(user1)
  except Exception as e:
    # 코드 실행 중 예외(에러)가 발생할 경우 에러 메시지를 출력합니다.
    print(f"Error: {e}")

# 스크립트가 메인으로 실행될 때 run() 함수를 호출합니다.
if __name__ == "__main__":
  run()
