import os
from settings import settings
import torch
from src.train_model import GPTModel
from src.step4 import test

def run(train_loader, last_model_name):
  """
  프로그램의 진입점: 장치 설정, 모델 로드, 학습 시작을 총괄
  """
  # 1. GPU 사용 가능 여부 확인 후 장치 할당
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # 2. 난수 시드 고정: 매번 동일한 초기화와 무작위성을 유지하기 위함 (재현성)
  torch.manual_seed(123)

  # 3. 모델 정의 및 기존 가중치 로드
  model = GPTModel()
  # weights_only=True: 보안 및 효율성을 위해 모델의 가중치 데이터만 로드
  model.load_state_dict(torch.load(os.path.join(settings.model_dir, last_model_name), map_location=device, weights_only=True))
  model.to(device) # 모델을 연산 장치(GPU/CPU)로 이동
  
  # 4. 최적화 알고리즘 설정: AdamW 사용 (학습률 0.0004, 가중치 감쇠 0.1)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
  
  # 5. 추가 학습 시작 (여기서는 1 에폭만 수행하도록 설정됨)
  test(model, optimizer, train_loader, device, 1)
