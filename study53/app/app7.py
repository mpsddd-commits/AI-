import gensim 
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
  print("="*100)

  return vocab_size, max_len, word_to_index, X_train, Y_train

# 사전 학습된 임베딩을 사용하는 모델
class PretrainedEmbeddingModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_len, embedding_matrix):
    super(PretrainedEmbeddingModel, self).__init__()
    # 1. 임베딩 층 생성
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # 2. 사전 학습된 가중치(embedding_matrix)를 임베딩 층에 덮어쓰기
    # nn.Parameter로 감싸야 학습 시 업데이트가 가능해집니다.
    self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    
    # 3. Fine-tuning 설정: True이면 우리 데이터에 맞춰 가중치가 미세하게 조정됨
    self.embedding.weight.requires_grad = True
    
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(embedding_dim * max_len, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    embedded = self.embedding(x)
    flattened = self.flatten(embedded)
    output = self.fc(flattened)
    return self.sigmoid(output)

# 임베딩 이식 및 학습 과정  
def step2(vocab_size, max_len, word_to_index, X_train, Y_train):
  # 1. 구글의 사전 학습된 모델 로드 (300차원)
  word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
  
  # 2. 우리 사전에 맞는 빈 임베딩 행렬 생성 (단어 수 x 300차원)
  embedding_matrix = np.zeros((vocab_size, 300))

  # 단어 검색 함수: 구글 모델에 단어가 있는지 확인
  def get_vector(word):
    return word2vec_model[word] if word in word2vec_model else None

  # 3. 우리 단어 사전의 각 단어에 대해 구글의 벡터값을 복사
  for word, i in word_to_index.items():
    if i > 1: # 특수 토큰(<PAD>, <UNK>)을 제외한 실제 단어들
      temp = get_vector(word)
      if temp is not None:
        embedding_matrix[i] = temp # 구글의 지식을 우리 행렬의 i번째 행에 저장

  # <PAD>나 <UNK>의 경우는 사전 훈련된 임베딩이 들어가지 않아서 0벡터임
  print('<PAD>나 <UNK> 경우 : ', embedding_matrix[0])

  # 학습 설정
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = PretrainedEmbeddingModel(vocab_size, 300, max_len, embedding_matrix).to(device)

  criterion = nn.BCELoss()
  optimizer = Adam(model.parameters())

  # 데이터 로더 준비
  train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(Y_train, dtype=torch.float32))
  train_dataloader = DataLoader(train_dataset, batch_size=2)
  print(f"데이터로더의 길이: {len(train_dataloader)}")

  # 4. 학습 루프
  print("학습 시작")
  for epoch in range(10):
    for inputs, targets in train_dataloader:
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(inputs).view(-1)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
  step2(*step1())
