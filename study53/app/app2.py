from gensim.models import Word2Vec
from konlpy.tag import Okt

def test(word):
  if word in model.wv:
    vector = model.wv[word]
    print(f"{word}의 벡터값: {vector}")
  else:
    print(f"'{word}'는 학습된 사전에 없습니다.")
  print("*"*100)

# 1. 데이터 준비 (코퍼스)
sentences = [
  "나는 자연어 처리를 배운다",
  "자연어 처리는 너무 재밌다",
  "딥러닝을 이용한 자연어 처리 성능이 좋다",
  "왕과 여왕은 궁전에 산다",
  "남자와 여자는 사람이다"
]

# 2. 형태소 분석기를 이용한 토큰화
okt = Okt()
tokenized_data = [okt.morphs(sentence) for sentence in sentences]
print(f"토큰화된 데이터:")
for data in tokenized_data:
  print(data)
print("="*100)

# 3. Word2Vec 모델 학습
# vector_size: 임베딩 된 벡터의 차원 (보통 100~300)
# window: 컨텍스트 윈도우 크기 (앞뒤로 몇 개 단어를 볼 것인지) - 유사도를 찾기 위해 쓰임
# min_count: 최소 등장 빈도수 - 제일 가까운거 찾기
# sg: 0이면 CBOW(Continuous Bag of Words) [중간에 있는 중심 단어를 예측], 1이면 Skip-gram [주변 단어들을 예측]
model = Word2Vec(sentences=tokenized_data, vector_size=10, window=5, min_count=1, sg=0)

size = 3
keyword = "왕"

# 4. 결과 확인
# '자연어'라는 단어의 밀집 벡터 값 확인
word_vector = model.wv[keyword]
print(f"'{keyword}'의 임베딩 벡터값: {word_vector}")
print("="*100)

# 5. 유사한 단어 찾기
similar_words = model.wv.most_similar(keyword, topn=size)
print(f"'{keyword}'와 유사한 단어들: {size}개")
for word, score in similar_words:
  if score > 0.5:
    print(f"{word}의 score: {score:.4f}")
    test(word)
