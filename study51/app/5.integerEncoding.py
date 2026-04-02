from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk import FreqDist
import numpy as np

# 빈도수가 높은 상위 n개의 단어만 추출하여 출력하는 함수
def getFrequency(vocab, vocab_size: int = 5):
  # most_common()은 빈도수가 높은 순으로 리스트를 반환
  vocab = vocab.most_common(vocab_size)
  print(f"빈도수가 높은 상위 5개의 단어: {vocab}")
  return vocab

# 분석할 원문 데이터
raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화: 텍스트를 문장 단위로 분리
sentences = sent_tokenize(raw_text)

# 단어와 빈도수를 저장할 딕셔너리
vocab = {} 

# 전처리가 끝난 문장들을 담을 리스트
preprocessed_sentences = [] 

# 영어 불용어 집합
stop_words = set(stopwords.words('english')) 

# --- [전처리 단계] ---
def step0():
  global vocab, preprocessed_sentences
  print("[전처리 단계]")
  for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence) # 단어 토큰화
    result = []

    for word in tokenized_sentence: 
      word = word.lower() # 1) 소문자화하여 단어 개수 줄이기
      if word not in stop_words: # 2) 불용어 제거
        if len(word) > 2: # 3) 길이가 짧은 단어 제거 (단어의 의미가 적은 경우가 많음)
          result.append(word)
          if word not in vocab:
            vocab[word] = 0 
          vocab[word] += 1
    preprocessed_sentences.append(result) 
  print(f"전처리된 문장: {preprocessed_sentences}")
  print("="*200)  

  # 단어 집합(Vocab) 출력 및 특정 단어 빈도 확인
  print('단어 집합 :', vocab)
  print("="*200)

# --- [방법 1: Python 기본 딕셔너리 활용 인코딩] ---
def step1():
  print("[방법 1: Python 기본 딕셔너리 활용 인코딩]")
  # 빈도수가 높은 순서대로 정렬
  vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
  print(f"빈도수가 높은 순서대로 정렬: {vocab_sorted}")

  word_to_index = {}
  i = 0
  for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 빈도수가 2번 이상인 단어만 인덱스 부여
      i = i + 1
      word_to_index[word] = i
  print(f"빈도수가 2번 이상인 단어: {word_to_index}")

  # 상위 5개(vocab_size) 단어만 남기고 나머지는 삭제
  vocab_size = 5
  words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]
  for w in words_frequency:
    del word_to_index[w]
  print(f"상위 5개(vocab_size) 단어: {word_to_index}")

  # OOV(Out-Of-Vocabulary) 처리: 단어 집합에 없는 단어를 위해 새로운 인덱스 부여
  word_to_index['OOV'] = len(word_to_index) + 1
  print(f"단어 집합에 없는 단어는 OOV(Out-Of-Vocabulary) 처리: {word_to_index}")

  # 정수 인코딩 진행: 단어를 정수 인덱스로 변환
  encoded_sentences = []
  encoded_words = []
  for sentence in preprocessed_sentences:
    encoded_word = []
    encoded_sentence = []
    for word in sentence:
      try:
        encoded_word.append(word)
        encoded_sentence.append(word_to_index[word])
      except KeyError:
        # 단어 집합에 없는 단어는 'OOV' 인덱스로 대체
        encoded_word.append('OOV')
        encoded_sentence.append(word_to_index['OOV'])
    encoded_words.append(encoded_word)
    encoded_sentences.append(encoded_sentence)
  print(f"정수 인코딩 문자: {encoded_words}")
  print(f"정수 인코딩 결과: {encoded_sentences}")
  print("="*200)

# --- [방법 2: collections.Counter 활용] ---
def step2():
  print("[방법 2: collections.Counter 활용]")
  # 모든 문장의 단어들을 하나의 리스트로 통합
  all_words_list = sum(preprocessed_sentences, [])
  # 리스트 내 단어 빈도수를 자동으로 계산
  vocab = Counter(all_words_list)
  print(f"barber라는 단어의 빈도수 : {vocab['barber']}")

  # 상위 5개 추출 및 인덱스 부여
  vocab = getFrequency(vocab, 5)
  word_to_index = {}
  i = 0
  for (word, frequency) in vocab :
    i = i + 1
    word_to_index[word] = i
  print(f"상위 5개 인덱스 부여: {word_to_index}")
  print("="*200)

# --- [방법 3: NLTK FreqDist 활용] ---
def step3():
  print("[방법 3: NLTK FreqDist 활용]")
  # np.hstack을 사용하여 2차원 리스트를 1차원으로 풀어서 전달
  vocab = FreqDist(np.hstack(preprocessed_sentences))
  print(f"barber라는 단어의 빈도수 : {vocab['barber']}")

  # 상위 5개 추출 및 딕셔너리 컴프리헨션으로 인덱싱 (enumerate 활용)
  vocab = getFrequency(vocab, 5)
  word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
  print(f"상위 5개 인덱스 부여: {word_to_index}")

if __name__ == '__main__':
  print(f"[원문 데이터]\n{raw_text}")
  print("="*200)
  print(f"[토큰화된 문장]\n{sentences}")
  print("="*200)
  step0()
  # step1()
  # step2()
  step3()
