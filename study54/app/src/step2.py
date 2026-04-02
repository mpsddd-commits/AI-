import tiktoken  # OpenAI에서 제공하는 BPE(Byte Pair Encoding) 기반 토큰화 라이브러리

def test1(text):
  """
  전체 문장을 한꺼번에 인코딩하고, 각 토큰 ID가 어떤 문자열에 대응하는지 상세히 분석하는 함수
  """
  # gpt2 모델이 사용하는 인코딩 방식(Encoding)을 가져옵니다.
  tokenizer = tiktoken.get_encoding("gpt2")
  
  # 입력받은 전체 텍스트를 토큰 ID의 리스트로 변환합니다. (Encoding)
  tokens = tokenizer.encode(text)
  
  # 원본 글자 수와 생성된 토큰의 개수를 비교 출력합니다.
  print("글자수:", len(text), "토큰수", len(tokens))
  
  # 변환된 토큰 ID 리스트 자체를 출력합니다.
  print(f"Tokens: {tokens}")
  
  # 토큰 ID 리스트를 다시 원래의 문장으로 복원하여 출력합니다. (Decoding)
  print(f"Decoded: {tokenizer.decode(tokens)}")
  
  # 각 토큰 ID를 하나씩 순회하며, 개별 ID가 어떤 단어/문자 조각을 의미하는지 매핑하여 출력합니다.
  for token in tokens:
    # 단일 토큰 ID를 리스트 형태로 전달하여 개별 디코딩 수행
    print(f"{token}\t -> {tokenizer.decode([token])}")
  
  print("="*100)

def test2(text):
  """
  텍스트를 한 글자(Character)씩 쪼개어 각각 인코딩했을 때의 결과를 확인하는 함수
  (토큰화가 단어 조각 단위로 어떻게 일어나는지 비교하기 위함)
  """
  tokenizer = tiktoken.get_encoding("gpt2")
  
  # 입력된 문자열의 각 문자(char)를 하나씩 반복 처리
  for char in text:
    # 각 문자를 토큰 ID로 변환 (한 글자라도 여러 개의 토큰으로 쪼개질 수 있음)
    token_ids = tokenizer.encode(char)
    
    # 변환된 토큰 ID를 다시 문자로 복원
    decoded = tokenizer.decode(token_ids)
    
    # [원본 문자] -> [토큰 ID 리스트] -> [복원된 문자] 순으로 출력
    print(f"{char} -> {token_ids} -> {decoded}")
      
  print("="*100)

def run():
  """
  테스트를 실행하는 메인 함수
  """
  text = "Harry Potter was a wizard."
  
  # 1. 문장 전체를 토큰화하는 방식 테스트
  test1(text)
  
  # 2. 문자를 하나씩 쪼개어 토큰화하는 방식 테스트
  test2(text)
