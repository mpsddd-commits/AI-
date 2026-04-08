from settings import settings
# Hugging Face의 데이터셋 처리 및 업로드를 위한 라이브러리를 불러옵니다.
from datasets import load_dataset

def run():
  """
  로컬의 데이터 파일을 읽어 Hugging Face Hub에 업로드하는 메인 함수입니다.
  """
  # 업로드 시 기록될 커밋 메시지를 설정합니다.
  commit_message = "첫 번째 업로드"
  
  # 1. 로컬 데이터 로드
  # "json" 형식의 파일을 읽어오며, 경로는 settings.json_file에 정의된 값을 사용합니다.
  # 이 과정에서 데이터는 Hugging Face의 Dataset 객체 형태로 변환됩니다.
  dataset = load_dataset("json", data_files=settings.json_file)
  
  # 2. Hugging Face Hub로 업로드
  # push_to_hub 함수를 사용하여 서버에 데이터를 전송합니다.
  dataset.push_to_hub(
    repo_id=settings.repo_name,   # 업로드할 저장소 이름 (예: "사용자ID/데이터셋이름")
    commit_message=commit_message, # 버전 관리를 위한 설명 메시지
    token=settings.hf_token       # Hugging Face API 접근을 위한 쓰기(Write) 권한 토큰
  )

# 스크립트가 메인으로 실행될 때만 run() 함수를 호출합니다.
if __name__ == "__main__":
  run()
