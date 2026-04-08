import torch
import os
from datetime import datetime
from safetensors.torch import save_file
from src.train_model import GPTModel
from settings import settings

def run(last_model_name, version):
  # 1. 장치 설정
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # 2. 모델 구조 정의 (제공된 train_model.py의 구조와 일치해야 함)
  model = GPTModel() 
  
  # 3. .pth 가중치 로드
  # weights_only=True를 사용하여 안전하게 로드합니다.
  state_dict = torch.load(os.path.join(settings.model_dir, last_model_name), map_location=device, weights_only=True)
  
  # 만약 state_dict만 저장된 것이 아니라 전체 모델이 저장된 경우를 대비해 체크
  if not isinstance(state_dict, dict):
    state_dict = state_dict.state_dict()
      
  model.load_state_dict(state_dict)
  model.eval() # 추론 모드로 설정

  newFolder = os.path.join(settings.model_dir, datetime.now().strftime("%Y%m%d_%H%M"))
  if not os.path.exists( newFolder ):
    os.makedirs(newFolder)

  # 4. safetensors로 저장
  # model.state_dict()를 직접 전달합니다.
  save_name = str(version).zfill(3) + ".safetensors"
  save_file(model.state_dict(), os.path.join(newFolder, save_name))
  print(f"변환 완료: {save_name}")
