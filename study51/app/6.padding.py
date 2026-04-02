import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 전처리가 완료된 단어 리스트 형태의 데이터
preprocessed_sentences = [
  ['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], 
  ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], 
  ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], 
  ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], 
  ['barber', 'went', 'huge', 'mountain']
]

# Keras Tokenizer를 생성하고 단어 집합(Vocabulary) 구축
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)

# 결과를 저장할 전역 변수
padded_np = None

def step1():
  """
  [Numpy를 이용한 수동 패딩]
  패딩의 원리를 이해하기 위해 직접 코드로 구현하는 단계입니다.
  """
  global padded_np
  # 1. 단어를 정수 인덱스로 변환 (Integer Encoding)
  encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
  print("정수 인코딩 결과:\n", encoded)

  # 2. 모든 문장 중 가장 긴 문장의 길이를 확인
  max_len = max(len(item) for item in encoded)
  print('최대 길이 :', max_len)

  # 3. 제로 패딩(Zero Padding): 모든 문장의 길이를 max_len에 맞춰 뒤에 0을 추가
  for sentence in encoded:
    while len(sentence) < max_len:
      sentence.append(0)

  # 4. Numpy 배열로 변환하여 행렬 구조 확인
  padded_np = np.array(encoded)
  print("Numpy 수동 패딩 결과:\n", padded_np)
  return padded_np

def step2():
  """
  [Keras pad_sequences를 이용한 패딩]
  실제 프로젝트에서 주로 사용하는 편리한 도구와 옵션을 실습합니다.
  """
  # 원본 정수 인코딩 데이터 준비
  encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
  
  # 1. 기본 패딩 (앞쪽에 0을 채움: pre padding)
  padded = pad_sequences(encoded)
  print("Keras 기본 패딩(앞쪽 0):\n", padded)

  # 2. 뒤쪽에 0 채우기 (padding='post')
  padded = pad_sequences(encoded, padding='post')
  print("Keras 뒤쪽 패딩(post):\n", padded)
  
  # Numpy로 직접 만든 결과와 Keras 결과가 일치하는지 확인
  print("Numpy 결과와 일치 여부:", (padded == padded_np).all())

  # 3. 최대 길이 제한 (maxlen)
  # 문장 길이를 5로 고정. 5보다 짧으면 0을 채우고, 길면 앞에서부터 자름.
  padded = pad_sequences(encoded, padding='post', maxlen=5)
  print("길이 5로 제한 패딩:\n", padded)

  # 4. 자르는 위치 변경 (truncating='post')
  # maxlen보다 길어서 단어를 버려야 할 때, 뒷부분을 삭제함.
  padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
  print("길이 5 제한 + 뒤쪽 삭제:\n", padded)

  # 참고: 패딩용 값 확인
  # 보통 0번 인덱스는 패딩용으로 예약되어 있으므로 실제 단어 집합 크기에 +1을 함.
  last_value = len(tokenizer.word_index) + 1 
  print("단어 집합 크기 + 1(패딩용 포함):", last_value)

if __name__ == '__main__':
  step1()
  print("="*100)
  step2()
