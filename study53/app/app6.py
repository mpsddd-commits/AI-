import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# 데이터 전처리 단계
def step1():
  # 학습용 문장과 정답(긍정: 1, 부정: 0) 레이블 정의
  sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
  y_train = [1, 0, 0, 1, 1, 0, 1]

  # 1. 토큰화: 공백 기준으로 단어 분리
  tokenized_sentences = [sent.split() for sent in sentences]
  print('단어 토큰화 된 결과 :', tokenized_sentences)

  # 2. 단어 집합 생성 및 빈도 파악
  word_list = []
  for sent in tokenized_sentences:
    for word in sent:
      word_list.append(word)

  word_counts = Counter(word_list)
  print('총 단어수 :', len(word_counts))
  # 빈도수 순으로 정렬된 리스트 생성
  vocab = sorted(word_counts, key=word_counts.get, reverse=True)
  print('등장 빈도순 :', vocab)

  # 3. 단어 사전 구축 (0: 패딩, 1: 모르는 단어, 2~: 실제 단어)
  word_to_index = {'<PAD>': 0, '<UNK>': 1}
  for index, word in enumerate(vocab) :
    word_to_index[word] = index + 2

  vocab_size = len(word_to_index)
  print('패딩 토큰, UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)

  # 4. 정수 인코딩: 문장을 인덱스 번호의 리스트로 변환
  def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sent in tokenized_X_data:
      index_sequences = []
      for word in sent:
        index_sequences.append(word_to_index.get(word, 1)) # 없으면 <UNK>인 1 반환
      encoded_X_data.append(index_sequences)
    return encoded_X_data

  X_encoded = texts_to_sequences(tokenized_sentences, word_to_index)
  print('정수 인코딩 결과 :', X_encoded)

  # 5. 패딩: 모든 문장의 길이를 가장 긴 문장(4)에 맞춤
  max_len = max(len(l) for l in X_encoded)
  print('최대 길이 :',max_len)

  def pad_sequences(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
      if len(sentence) != 0:
        features[index, :len(sentence)] = np.array(sentence)[:max_len]
    return features

  X_train = pad_sequences(X_encoded, max_len)
  Y_train = np.array(y_train)

  print('패딩 결과 :', X_train)
  print("="*100)

  return vocab_size, max_len, X_train, Y_train

# 신경망 모델 정의
class SimpleModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_len):
    super(SimpleModel, self).__init__()
    # 임베딩 층: 정수 인덱스를 고정된 크기(embedding_dim)의 밀집 벡터로 변환
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # Flatten: (문장길이, 임베딩차원)의 2차원 데이터를 1차원으로 쭉 펼침 (연산을 위함)
    self.flatten = nn.Flatten()
    
    # 선형 층: 펼쳐진 벡터를 입력받아 최종 점수(1개)를 계산
    self.fc = nn.Linear(embedding_dim * max_len, 1)
    
    # 시그모이드: 출력값을 0과 1 사이(확률값)로 변환
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    embedded = self.embedding(x)      # [Batch, Max_len, Embedding_dim]
    flattened = self.flatten(embedded) # [Batch, Max_len * Embedding_dim]
    output = self.fc(flattened)        # [Batch, 1]
    return self.sigmoid(output)

# 학습 단계
def step2(vocab_size, max_len, X_train, Y_train):
  # 장치 설정 (GPU 사용 가능하면 cuda, 아니면 cpu)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  embedding_dim = 100 # 단어 하나를 100개의 숫자로 표현
  simple_model = SimpleModel(vocab_size, embedding_dim, max_len).to(device)

  # 손실 함수 (이진 분류용 Binary Cross Entropy 사용)
  criterion = nn.BCELoss()
  # 최적화 알고리즘 (Adam optimizer)
  optimizer = Adam(simple_model.parameters())

  # PyTorch 전용 데이터셋 및 데이터로더 생성 (배치 크기 2로 설정) - 교과서
  train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), 
                                torch.tensor(Y_train, dtype=torch.float32))
  train_dataloader = DataLoader(train_dataset, batch_size=2)
  print(f"데이터로더의 길이: {len(train_dataloader)}")

  # 전체 데이터를 10번 반복 학습(Epoch) - 학생에게 교육횟수
  print("학습 시작")
  for epoch in range(10):
    for inputs, targets in train_dataloader:
      inputs, targets = inputs.to(device), targets.to(device)
      
      # 1. 기울기 초기화
      optimizer.zero_grad()
      
      # 2. 모델 예측 (Forward)
      outputs = simple_model(inputs).view(-1)
      
      # 3. 오차 계산 (Loss) 0으로 가게 해야한다.
      loss = criterion(outputs, targets)
      
      # 4. 역전파 (Backward): 오차에 대한 기울기 계산
      loss.backward()
      
      # 5. 가중치 업데이트 (Step)
      optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
  step2(*step1())
