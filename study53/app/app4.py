import torch

# 1. 학습용 텍스트 데이터를 정의합니다.
train_data = 'you need to know how to code'

# 2. 중복을 제거한 단어 집합(Vocabulary)을 만듭니다.
word_set = set(train_data.split())

# 3. 각 단어에 고유한 정수 인덱스를 부여하는 사전을 만듭니다.
# i+2를 하는 이유는 0번과 1번 인덱스를 특수 토큰용으로 비워두기 위함입니다.
vocab = {word: i+2 for i, word in enumerate(word_set)}

# 4. 특수 토큰(Special Tokens)을 사전에 추가합니다.
# <unk>: Unknown, 사전에 없는 모르는 단어 처리용
# <pad>: Padding, 문장의 길이를 일정하게 맞추기 위해 채워넣는 빈칸용
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(f"완성된 사전: {vocab}")

# 5. 임베딩 테이블(Embedding Table)을 직접 생성합니다. (차원: 3)
# 각 행(Row)은 사전(vocab)의 인덱스 번호와 대응됩니다.
# 예: 0번 행은 <unk> (Unknown)의 벡터, 1번 행은 <pad> (Padding)의 벡터입니다.
data = [
  [ 0.0,  0.0,  0.0], # Index 0: <unk> 
  [ 0.0,  0.0,  0.0], # Index 1: <pad> 
  [ 0.2,  0.9,  0.3], # Index 2 이후는 단어들...
  [ 0.1,  0.5,  0.7],
  [ 0.2,  0.1,  0.8],
  [ 0.4,  0.1,  0.1],
  [ 0.1,  0.8,  0.9],
  [ 0.6,  0.1,  0.1]
]

embedding_table = torch.FloatTensor(data)

# 6. 새로운 문장('you need to run')을 입력받아 처리합니다.
sample = 'you need to run'.split()
indexs = []

# 7. 입력된 단어를 사전을 이용해 인덱스 번호로 변환합니다.
for word in sample:
  try:
    # 사전에 있는 단어라면 해당 인덱스를 추가
    indexs.append(vocab[word])
  except KeyError:
    # 'run'처럼 사전에 없는 단어는 <unk> 토큰의 인덱스(0)를 추가
    indexs.append(vocab['<unk>'])

# 파이토치 연산을 위해 리스트를 텐서(LongTensor)로 변환합니다.
indexs = torch.LongTensor(indexs)
print(f"대상 단어: {sample}\n인덱스: {indexs}")

# 8. 룩업(Lookup) 과정: 인덱스 번호를 이용해 임베딩 테이블에서 해당 행의 벡터를 추출합니다.
# 결과적으로 'you', 'need', 'to', 'run'에 해당하는 3차원 벡터 4개가 나옵니다.
lookup_result = embedding_table[indexs, :]

print("\n--- 룩업 결과 (단어 벡터) ---")
for word, vector in zip(sample, lookup_result):
  print(f"단어: {word:6} | 인덱스: {vocab.get(word, 0)} | 벡터: {vector.tolist()}")
