import gensim 

# 1. 구글에서 제공하는 사전 학습된(Pre-trained) Word2Vec 모델을 로드합니다.
# 'GoogleNews-vectors-negative300.bin.gz'는 300차원의 벡터로 학습된 바이너리 파일입니다.
# binary=True는 해당 파일이 이진 형식임을 알려줍니다.
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)

# 2. 모델의 전체 구조(Shape)를 출력합니다.
# 출력 예시: (3000000, 300) -> 300만 개의 단어가 각각 300차원의 벡터로 저장되어 있음.
print(f"모델의 전체 구조 : {word2vec_model.vectors.shape}")

# 3. .similarity('단어1', '단어2')를 사용하여 두 단어 사이의 코사인 유사도를 계산합니다.
# 'this'와 'is'의 유사도 (문법적 기능을 수행하는 단어들 간의 관계)
print(f"'this'와 'is'의 유사도 : {word2vec_model.similarity('this', 'is')}")

# 'post'(게시물/우편)와 'book'(책)의 유사도
print(f"'post'(게시물/우편)와 'book'(책)의 유사도 : {word2vec_model.similarity('post', 'book')}")

# 'post'(게시물)와 'movie'(영화)의 유사도
# 'book'보다 'movie'가 'post'와 더 비슷한 맥락에서 쓰였다면 점수가 더 높게 나옵니다.
print(f"'post'(게시물)와 'movie'(영화)의 유사도 : {word2vec_model.similarity('post', 'movie')}")

# 4. 'book'이라는 단어가 가진 실제 300개의 밀집 벡터(Dense Vector) 값을 출력합니다.
print(f"book 벡터값: {word2vec_model['book']}")
