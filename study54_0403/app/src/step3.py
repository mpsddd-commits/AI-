import os
from settings import settings
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 커스텀 데이터셋 클래스 정의
# PyTorch의 Dataset 클래스를 상속받아 텍스트 데이터를 모델 학습용으로 가공합니다.
class MyDataset(Dataset):
  def __init__(self, tokenizer, txt, max_length, stride):
    """
    Args:
        tokenizer: 텍스트를 토큰 ID로 변환할 토크나이저 (tiktoken)
        txt: 전체 입력 텍스트
        max_length: 모델이 한 번에 처리할 시퀀스의 길이 (윈도우 크기)
        stride: 다음 시퀀스를 추출하기 위해 이동하는 간격 (슬라이딩 윈도우)
    """
    self.input_ids = []
    self.target_ids = []
    
    # 전체 텍스트를 토큰 ID 리스트로 변환
    token_ids = tokenizer.encode(txt)
    print("# of tokens in txt:", len(token_ids))

    # 슬라이딩 윈도우 방식으로 데이터를 조각내어 저장
    # 마지막 max_length 범위를 벗어나지 않도록 반복문 수행
    for i in range(0, len(token_ids) - max_length, stride):
      # 모델의 입력값 (현재 위치 i부터 max_length 만큼)
      input_chunk = token_ids[i:i + max_length]
      # 모델의 정답값 (입력값보다 한 글자 뒤에 있는 토큰들)
      # 언어 모델은 '다음 토큰'을 예측하므로 1칸씩 밀린 데이터를 정답으로 사용함
      target_chunk = token_ids[i + 1: i + max_length + 1]
      
      # 리스트에 PyTorch 텐서(Tensor) 형태로 추가
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
      # 전체 데이터 샘플의 개수를 반환
      return len(self.input_ids)

  def __getitem__(self, idx):
      # 특정 인덱스(idx)에 해당하는 입력값과 정답값을 쌍으로 반환
      return self.input_ids[idx], self.target_ids[idx]

# 2. 실행 함수 정의
def run(token_size:int = 10000):
  datasets = []
  
  # target_dir 디렉토리에 있는 파일 목록을 하나씩 확인
  for filename in os.listdir(settings.target_dir):
    if filename.endswith(".txt"):
      print("파일명:", filename)
      
      # 텍스트 파일 읽기 (BOM 제거를 위해 utf-8-sig 사용)
      with open(os.path.join(settings.target_dir, filename), 'r', encoding='utf-8-sig') as file:
          txt = file.read()

      # 성능과 메모리를 고려하여 텍스트의 처음 일정 부분만 사용
      # txt = txt[:token_size]

      # GPT-2용 토크나이저 인코딩 로드
      tokenizer = tiktoken.get_encoding("gpt2") # BPE(Byte Pair Encoding)
      
      # 커스텀 데이터셋 생성
      # max_length=32: 한 번에 32개 토큰 학습 / stride=4: 4토큰씩 옆으로 이동하며 샘플링
      dataset = MyDataset(tokenizer, txt, max_length = 32, stride = 4)
      
      # DataLoader 생성: 데이터셋을 배치(batch) 단위로 묶고 순서를 섞음
      # batch_size=128: 한 번에 128개의 데이터를 모델에 전달
      # drop_last=True: 마지막 배치가 128개가 안 되면 버림 (학습 안정성)
      train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
      
      datasets.append(train_loader)
      
      # 데이터 로더가 잘 작동하는지 확인하기 위해 첫 번째 배치 데이터 추출
      dataiter = iter(train_loader)
      x, y = next(dataiter)

      # 첫 번째 샘플(x[0])을 디코딩하여 입력과 정답의 관계를 확인
      # 정답(Target)은 입력(Input)보다 항상 한 토큰씩 뒤에 있어야 함
      print(f"Input: {tokenizer.decode(x[0].tolist())}")
      print(f"Target: {tokenizer.decode(y[0].tolist())}")
          
  return datasets
