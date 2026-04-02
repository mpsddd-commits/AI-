import torch

# 1. 단어들을 원-핫 벡터(One-Hot Vector) 형태로 정의합니다.
# 각 단어는 전체 단어 집합의 크기(5)만큼의 길이를 가지며, 
# 자기 자신을 나타내는 인덱스만 1이고 나머지는 모두 0입니다.
dog      = torch.FloatTensor([1, 0, 0, 0, 0]) # '강아지'의 위치는 0번 인덱스
cat      = torch.FloatTensor([0, 1, 0, 0, 0]) # '고양이'의 위치는 1번 인덱스
computer = torch.FloatTensor([0, 0, 1, 0, 0]) # '컴퓨터'의 위치는 2번 인덱스
netbook  = torch.FloatTensor([0, 0, 0, 1, 0]) # '넷북'의 위치는 3번 인덱스
book     = torch.FloatTensor([0, 0, 0, 0, 1]) # '책'의 위치는 4번 인덱스

# 2. torch.cosine_similarity 함수를 사용하여 두 벡터 간의 코사인 유사도를 계산합니다.
# dim=0은 1차원 벡터의 방향을 따라 계산하라는 의미입니다.

# 강아지(dog)와 고양이(cat)의 유사도 측정
# 의미상으로는 비슷할 수 있지만, 원-핫 벡터상에서는 겹치는 인덱스가 없으므로 0.0이 나옵니다.
print(f"dog vs cat: {torch.cosine_similarity(dog, cat, dim=0)}")

# 고양이(cat)와 컴퓨터(computer)의 유사도 측정
# 전혀 다른 의미를 가진 단어들이며, 역시 유사도는 0.0입니다.
print(f"cat vs computer: {torch.cosine_similarity(cat, computer, dim=0)}")

# 컴퓨터(computer)와 넷북(netbook)의 유사도 측정
# 실제로는 매우 유사한 단어들이지만, 원-핫 인코딩은 이를 구분하지 못해 0.0이 나옵니다.
print(f"computer vs netbook: {torch.cosine_similarity(computer, netbook, dim=0)}")

# 넷북(netbook)과 책(book)의 유사도 측정
# 마찬가지로 유사도는 0.0이 나옵니다.
print(f"netbook vs book: {torch.cosine_similarity(netbook, book, dim=0)}")
