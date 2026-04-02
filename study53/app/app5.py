import torch.nn as nn

# 1. 학습용 텍스트 데이터 준비
train_data = 'you need to know how to code'

# 2. 중복을 제거한 단어 집합 생성
word_set = set(train_data.split())

# 3. 단어 사전(Vocabulary) 구축
# i+2를 통해 0번(<unk>), 1번(<pad>) 자리를 비워둡니다.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

# 4. nn.Embedding 레이어 생성
# num_embeddings: 사전의 크기 (총 단어 개수)
# embedding_dim: 각 단어를 표현할 벡터의 차원 (여기서는 3차원)
# padding_idx: 패딩 토큰의 인덱스를 지정 (이 인덱스의 벡터는 학습 시 업데이트되지 않고 0으로 유지됨)
embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3, padding_idx=1)

# 5. 임베딩 레이어의 가중치(Weight) 출력
# 이 가중치가 바로 '임베딩 테이블' 그 자체입니다.
# 처음 생성 시에는 내부적으로 무작위 숫자(Random Initialized)로 채워져 있습니다.
print(embedding_layer.weight)
