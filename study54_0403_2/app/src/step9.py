from transformers import AutoTokenizer
import os
from settings import settings

def run(last_model_name):

  save_directory = os.path.join(settings.model_dir, last_model_name)

  # 1. 저장할 폴더가 없으면 생성
  if not os.path.exists(save_directory):
      os.makedirs(save_directory)
      print(f"폴더 생성됨: {save_directory}")

  # 2. Hugging Face에서 표준 gpt2 토크나이저 로드
  # tiktoken의 gpt2와 어휘 구성이 동일합니다.
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  
  # 3. 모든 관련 파일을 해당 폴더에 저장
  tokenizer.save_pretrained(save_directory)
  
  print("-" * 30)
  print(f"저장된 파일 목록 (@ {save_directory}):")
  for file in os.listdir(save_directory):
    print(f" - {file}")
  print("-" * 30)