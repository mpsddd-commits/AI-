import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

# NLTK 리소스(토큰화 규칙, 불용어 리스트, 단어 사전 등)를 다운로드하는 함수
def download_nltk_data():
  nltk.download('punkt')      # 문장 및 단어 토큰화를 위한 데이터
  nltk.download('stopwords')  # a, the, is 같은 불용어 목록
  nltk.download('wordnet')    # 표제어 추출(Lemmatization)을 위한 사전 데이터

def getData():
  texts = []
  for filename in os.listdir("./data"):
    with open(os.path.join("./data", filename), 'r', encoding='utf-8') as file:
      book_text = file.read()
    
    # texts.append(book_text)/
    texts.append(book_text[:100])
  return texts

def step1(text: str):
  """
  Step 1: 텍스트 전처리 (Preprocessing)
  문장을 단어로 쪼개고, 의미 없는 단어 제거 및 단어의 원형을 추출합니다.
  """
  # 1. 문장 토큰화: 전체 텍스트를 문장 단위로 분리
  sentences = sent_tokenize(text)
  
  # 2. 단어 토큰화 및 소문자화: 첫 번째 문장을 단어 단위로 쪼개고 소문자로 변환
  tokens = word_tokenize(sentences[0].lower())
  
  # 3. 불용어(Stopwords) 제거 및 특수문자 제외: 'is', 'the' 등을 제거하고 알파벳/숫자만 남김
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
  
  # 4. 표제어 추출(Lemmatization): 단어의 기본 형태(am/are/is -> be 등)를 찾음
  lemmatizer = WordNetLemmatizer()
  final_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
  
  print(f"최종 토큰: {final_tokens}")
  return final_tokens

def step2(sentences):
  """
  Step 2: 정수 인코딩 (Integer Encoding)
  컴퓨터가 이해할 수 있도록 단어를 고유한 숫자(인덱스)로 변환합니다.
  """
  tokenizer = Tokenizer()
  # 단어 집합(Vocabulary) 생성
  tokenizer.fit_on_texts(sentences)
  # 단어를 정수 시퀀스로 변환 (예: 'harry' -> 1, 'potter' -> 2)
  sequences = tokenizer.texts_to_sequences(sentences)
  
  print("정수 인코딩 결과:", sequences)
  
  # 다음 단계인 패딩과 임베딩으로 전달
  padded_data = step3(sequences)
  step4(tokenizer, padded_data)

def step3(sequences):
  """
  Step 3: 패딩 (Padding)
  서로 다른 길이의 문장들을 가장 긴 문장 길이에 맞춰 길이를 통일합니다.
  """
  # padding='post'는 빈 공간을 뒤쪽(0으로) 채운다는 의미입니다.
  padded_data = pad_sequences(sequences, padding='post') 
  print("패딩 결과 (행렬):\n", padded_data)
  return padded_data

def step4(tokenizer, padded_data):
  """
  Step 4: 워드 임베딩 (Word Embedding)
  숫자로 된 단어 인덱스를 고차원의 밀집 벡터(실수값들의 묶음)로 변환합니다.
  """
  # 전체 단어 사전의 크기 (0번 패딩 토큰을 포함하기 위해 +1)
  vocab_size = len(tokenizer.word_index) + 1
  # 임베딩 벡터의 차원 (하나의 단어를 8개의 숫자로 표현)
  embedding_dim = 8
  
  model = Sequential()
  # Embedding 레이어: (단어 개수, 벡터 크기, 입력 문장 길이)
  # input_length=7은 입력되는 시퀀스의 길이를 고정하는 설정입니다.
  model.add(Embedding(vocab_size, embedding_dim, input_length=7))
  
  # 실제 데이터를 모델에 통과시켜 임베딩 결과를 예측(계산)
  output = model.predict(padded_data)
  
  # 결과 크기: (문장 개수, 문장 내 단어 개수, 임베딩 차원)
  print("임베딩 후 데이터 크기:", output.shape) 

if __name__ == "__main__":
  # NLTK 데이터가 없다면 먼저 실행하세요: download_nltk_data()
  for text in getData():
    print(text)

  # sentences = []
  # text = "Harry Potter was a highly unusual boy in many ways. He didn't like the holidays."
  
  # # 1. 전처리 수행 (첫 번째 문장만 처리됨)
  # processed_tokens = step1(text)
  # sentences.append(processed_tokens)
  
  # # 2. 인코딩 및 모델 흐름 시작
  # step2(sentences)
