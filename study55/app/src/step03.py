from settings import settings
# Hugging Face Hub 관리 도구 중 저장소 삭제를 위한 함수를 불러옵니다.
from huggingface_hub import delete_repo

def run():
  """
  설정된 정보에 따라 Hugging Face Hub의 특정 저장소를 삭제하는 함수입니다.
  """
  # delete_repo 함수를 호출하여 실제 삭제를 진행합니다.
  delete_repo(
    repo_id=settings.repo_name, # 삭제할 저장소의 ID (예: '사용자ID/데이터셋이름')
    repo_type="dataset",        # 삭제하려는 저장소의 유형을 'dataset'으로 명시합니다. (model, space 등 가능)
    token=settings.hf_token     # 삭제 권한을 확인하기 위한 Hugging Face Access Token입니다.
  )
  # 삭제 작업은 별도의 경고 없이 즉시 실행되므로 실행 전 repo_id를 반드시 확인해야 합니다.

# 스크립트가 직접 실행될 경우에만 run() 함수를 동작시킵니다.
if __name__ == "__main__":
  run()
