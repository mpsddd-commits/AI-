from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
import kss
from konlpy.tag import Okt
from konlpy.tag import Kkma
import argparse # CLI에서 파라메터 던지기 위한 것

def step0():
  """
  영어 단어 토큰화(Word Tokenization) 비교
  - 다양한 라이브러리가 구두점(., !), 줄임표(Don't)를 어떻게 처리하는지 확인
  """
  txt = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
  
  # 1. word_tokenize: NLTK의 기본 토큰화. "Don't"를 "Do"와 "n't"로 분리
  print('단어 토큰화1 :', word_tokenize(txt))
  
  # 2. WordPunctTokenizer: 구두점을 별도로 분리하는 특징이 있음. "Don't"를 "Don", "'", "t"로 분리
  print('단어 토큰화2 :', WordPunctTokenizer().tokenize(txt))
  
  # 3. text_to_word_sequence: 케라스(TensorFlow) 제공. 모든 문자를 소문자로 바꾸고 구두점(마침표, 컴마 등)을 제거
  print('단어 토큰화3 :', text_to_word_sequence(txt))

def step1():
  """
  영어 문장 토큰화(Sentence Tokenization)
  - 마침표(.)가 단순히 약어(Ph.D.)에 쓰인 것인지 문장의 끝인지 구분하는 능력을 확인
  """
  txt1 = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
  txt2 = "I am actively looking for Ph.D. students. and you are a Ph.D student."
  
  # sent_tokenize: NLTK에서 제공하는 영어 문장 분리 도구
  print('문장 토큰화1 :', sent_tokenize(txt1))

  # Ph.D. 같은 약어를 인식하여 문장 중간에서 끊기지 않도록 처리
  print('문장 토큰화2 :', sent_tokenize(txt2))

def step2():
  """
  한국어 문장 토큰화 (KSS - Korean Sentence Splitter)
  - 한국어는 마침표 외에도 문장 분리 기준이 복잡하여 전용 라이브러리 사용 권장
  """
  txt = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'

  # kss.split_sentences: 한국어 특성을 반영하여 문장 단위로 분리
  print('한국어 문장 토큰화 :', kss.split_sentences(txt))

def step3():
  """
  영어 품사 태깅(Part-of-Speech Tagging)
  - 단어의 역할(명사, 동사, 형용사 등)을 정의된 태그로 표시
  """
  txt = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
  
  # 먼저 단어 토큰화를 수행
  tokenized_sentence = word_tokenize(txt)
  print('단어 토큰화 :', tokenized_sentence)

  # pos_tag: NLTK 제공. PRP(인칭 대명사), VBP(동사), RB(부사) 등의 태그를 붙임
  print('품사 태깅 :', pos_tag(tokenized_sentence))

def step4():
  """
  한국어 형태소 분석 1 (Okt - Open Korean Text)
  - 트위터에서 개발된 분석기로, 속도가 빠르고 정규화 기능이 강점
  """
  okt = Okt()
  txt = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"

  # morphs: 형태소(Morpheme) 단위로 분리
  print('OKT 형태소 분석 :', okt.morphs(txt))

  # pos: 형태소 분리 후 품사 태깅 (예: 명사, 동사, 조사 등)
  print('OKT 품사 태깅 :', okt.pos(txt))

  # nouns: 문장에서 명사(Noun)만 추출
  print('OKT 명사 추출 :', okt.nouns(txt))

def step5():
  """
  한국어 형태소 분석 2 (Kkma - 꼬꼬마)
  - 서울대학교에서 개발. 분석이 정밀하지만 Okt에 비해 속도가 느린 편
  """
  kkma = Kkma()
  txt = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"

  # morphs: 형태소 단위로 분리 (Okt와 분석 결과가 다를 수 있음)
  print('꼬꼬마 형태소 분석 :', kkma.morphs(txt))

  # pos: 상세한 품사 태깅 (예: NNG 일반명사, JKS 주격조사 등 상세 분류)
  print('꼬꼬마 품사 태깅 :', kkma.pos(txt))

  # nouns: 명사만 추출
  print('꼬꼬마 명사 추출 :', kkma.nouns(txt))

if __name__ == '__main__':
  # 터미널(CLI)에서 -s 인자를 받아 실행할 단계를 결정
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', type=int, help="실행할 단계 선택 (step1~step5, 미입력 시 step0)")
  
  try:
    args = parser.parse_args()
    
    # 입력된 step 값에 따라 해당 함수 실행
    if args.s == 1:
      step1()
    elif args.s == 2:
      step2()
    elif args.s == 3:
      step3() 
    elif args.s == 4:
      step4() 
    elif args.s == 5:
      step5() 
    else:
      # 기본값 또는 step0 입력 시 실행
      step0()
  except SystemExit:
    # argparse 오류 시(잘못된 인자 입력 등) 프로그램 종료 처리
    print("잘못된 인자 입력이 발생했습니다.")
    exit()
