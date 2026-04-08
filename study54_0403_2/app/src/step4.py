import os
import time
from datetime import datetime
from settings import settings
import torch
from src.train_model import GPTModel
import matplotlib.pyplot as plt

# 1. 데코레이터: 함수의 실행 시간을 측정하기 위함
def clock(func):
  def clocked(*args, **kwargs):
    start = time.perf_counter()  # 시작 시간 기록
    result = func(*args, **kwargs)
    end = time.perf_counter()    # 종료 시간 기록
    print(f"[{func.__name__}] 실행 시간: {end - start:.6f}s")
    return result
  return clocked

# 2. 손실 그래프 출력 함수
def view_loss_curve(losses):
  plt.plot(losses)
  plt.xlabel("Epoch")      # X축: 에폭
  plt.ylabel("Loss")       # Y축: 손실값
  plt.title("Training Loss Curve")
  plt.show()

# 3. 단일 에폭(Epoch) 학습 함수
@clock  # 실행 시간 측정을 위해 데코레이터 적용
def epoch_run(model, optimizer, train_loader, device, tokens_seen, global_step):
  epoch_loss = 0
  for input_batch, target_batch in train_loader:
    # 기울기 초기화 (이전 배치의 계산 결과가 남지 않도록)
    optimizer.zero_grad()
    
    # 데이터를 GPU(cuda) 혹은 CPU로 이동
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 모델 예측 (Forward Pass)
    logits = model(input_batch)
    
    # 손실 계산 (Cross Entropy)
    # flatten(0, 1)을 통해 [배치, 시퀀스, 토큰수] -> [배치*시퀀스, 토큰수] 형태로 변환하여 계산
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    
    epoch_loss += loss.item() # 시각화를 위해 손실값 누적
    
    # 역전파: 기울기 계산 (Backward Pass)
    loss.backward()
    
    # 가중치 업데이트 (Optimization Step)
    optimizer.step()
    
    # 학습에 사용된 토큰 수 및 전체 스텝 수 카운트
    tokens_seen += input_batch.numel()
    global_step += 1

    # 특정 스텝마다 로그 출력
    if global_step % 1000 == 0:
      print(f"Tokens seen: {tokens_seen}")
          
  return epoch_loss, tokens_seen, global_step

# 4. 전체 테스트/학습 루프 관리 함수
def test(model, optimizer, train_loader, device, num_epochs:int = 10):
  tokens_seen, global_step = 0, -1
  losses = []
  newFolder = os.path.join(settings.model_dir, datetime.now().strftime("%Y%m%d_%H%M"))

  if not os.path.exists( newFolder ):
    os.makedirs(newFolder)

  for epoch in range(num_epochs):
    model.train() # 모델을 학습 모드로 설정 (드롭아웃, 배치 정규화 등에 영향)
    
    # 1에폭 학습 실행
    epoch_loss, tokens_seen, global_step = epoch_run(model, optimizer, train_loader, device, tokens_seen, global_step)
    
    # 평균 손실값 계산 및 저장
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
    
    # 각 에폭 종료 후 모델 가중치 저장 (파일명 예: 001.pth, 002.pth)
    save_name = str(epoch + 1).zfill(3) + ".pth"
    torch.save(model.state_dict(), os.path.join(newFolder, save_name))

  # 모든 에폭이 끝나면 학습 곡선 출력
  # view_loss_curve(losses)

# 5. 실행 시작 함수
def run(train_loader):
  # 가속 장치 설정 (NVIDIA GPU 사용 가능 여부 확인)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # 실험의 재현성을 위해 랜덤 시드 고정
  torch.manual_seed(123)
  
  # 모델 생성 및 장치(GPU/CPU)로 전송
  model = GPTModel()
  model.to(device)
  
  # 옵티마이저 설정 (AdamW: 가중치 감쇠가 포함된 Adam 변형 버전)
  # lr: 학습률, weight_decay: 가중치가 너무 커지지 않게 규제
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

  # 실제 학습 시작
  test(model, optimizer, train_loader, device, 3)
