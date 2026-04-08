import re  # 정규 표현식(Regular Expression) 사용을 위한 모듈
import os  # 파일 경로 및 디렉토리 관리를 위한 모듈
from settings import settings  # 외부 설정 파일(settings.py)에서 설정값들을 가져옴

def clean_text(filename):
  """
  파일을 읽어 텍스트를 정제(공백/줄바꿈 제거)한 후 저장하는 함수
  """
  # 1. 파일 읽기: settings에 정의된 입력 디렉토리와 파일명을 합쳐서 파일을 엽니다.
  with open(os.path.join(settings.input_dir, filename), 'r', encoding='utf-8') as file:
    book_text = file.read()

  # 2. 텍스트 정제 (정규 표현식 사용)
  # r'\n+' : 하나 이상의 연속된 줄바꿈 문자를 찾아 공백(' ') 하나로 치환
  cleaned_text = re.sub(r'\n+', ' ', book_text)
  
  # r'\s+' : 하나 이상의 연속된 모든 공백 문자(스페이스, 탭 등)를 공백 하나로 치환
  # 이 과정을 통해 문장 사이의 불필요한 간격이 정렬됩니다.
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

  # 3. 결과 출력: 처리된 파일의 이름(접두사 포함)과 총 글자 수를 콘솔에 표시
  print(settings.prefix_name + filename, len(cleaned_text), "characters")

  # 4. 파일 저장: 정제된 텍스트를 대상(target) 디렉토리에 새로운 이름으로 저장
  output_path = os.path.join(settings.target_dir, settings.prefix_name + filename)
  with open(output_path, 'w', encoding='utf-8') as file:
    file.write(cleaned_text)

def run():
  """
  입력 디렉토리 내의 파일들을 탐색하여 조건에 맞는 파일만 처리하는 실행 함수
  """
  # 설정된 입력 디렉토리 내의 모든 파일 목록을 가져와서 반복문 수행
  for filename in os.listdir(settings.input_dir):
      
    # 파일 확장자가 ".txt"인 경우에만 진행
    if filename.endswith(".txt"):
        
      # 파일명이 "Book1"으로 시작하는 파일만 골라냄
      if filename.startswith("Book1"):
        print("파일명:", filename)
        
        # 정제 함수 호출
        clean_text(filename)
        
        # 'break'가 있으므로 조건에 맞는 파일 1개만 처리하고 루프를 종료합니다.
        # 모든 Book1 파일을 처리하려면 이 break를 제거하거나 수정해야 합니다.
        break