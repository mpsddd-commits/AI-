from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# 1. 기초적인 원-핫 인코딩 구현 (Pure Python)
print("[1. 기초적인 원-핫 인코딩]")
okt = Okt()  
# 한국어 문장을 형태소 단위로 토큰화
tokens = okt.morphs("나는 자연어 처리를 배운다")  
print(tokens) # 결과: ['나', '는', '자연어', '처리', '를', '배운다']

# 각 토큰에 고유한 정수 인덱스를 부여하여 단어 집합(Vocabulary) 생성
word_to_index = {word : index for index, word in enumerate(tokens)}
print('단어 집합 :', word_to_index)

def one_hot_encoding(word, word_to_index):
  """
  특정 단어를 원-핫 벡터로 변환하는 함수
  - 단어 집합의 크기만큼 0으로 채워진 리스트 생성
  - 해당 단어의 인덱스 위치만 1로 변경
  """
  size = len(word_to_index)
  one_hot_vector = [0] * size
  index = word_to_index[word]
  print(f"전체 길이: {size}  > 위치값: {index}")
  one_hot_vector[index] = 1
  return one_hot_vector

# "자연어"라는 단어의 원-핫 벡터 출력
print(one_hot_encoding("자연어", word_to_index))

print("="*100)

# 2. 케라스(Keras)를 이용한 원-핫 인코딩
print("[2. 케라스(Keras)를 이용한 원-핫 인코딩]")
text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

# Tokenizer 객체 생성 및 빈도수 기반 단어 집합 구축
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
# 생성된 단어 집합 확인 (빈도수가 높을수록 낮은 인덱스 부여)
print('단어 집합 :', tokenizer.word_index)

# 새로운 문장을 정수 시퀀스로 변환
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print('정수 인코딩 결과 :', encoded)

# to_categorical(): 정수 시퀀스를 원-핫 벡터의 행렬로 변환
# 결과는 (단어 개수, 단어 집합의 크기 + 1)의 형태를 가짐
# (+1이 되는 이유는 인덱스 0번을 사용하지 않기 때문)
one_hot = to_categorical(encoded)
print('케라스 원-핫 인코딩 결과 :\n', one_hot)
