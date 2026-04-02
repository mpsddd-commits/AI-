from nltk.stem import WordNetLemmatizer

# WordNetLemmatizer 객체 생성
# NLTK에서 제공하는 표제어 추출기로, 영어 단어의 형태소 분석을 통해 기본형을 찾습니다.
lemmatizer = WordNetLemmatizer()

# 테스트할 단어 리스트 정의
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :', words)

# 리스트 컴프리헨션을 사용하여 각 단어의 표제어(Lemma)를 추출
# 기본적으로 lemmatize()는 단어를 '명사'로 간주하고 처리합니다.
# 결과 예: 'lives' -> 'life', 'has' -> 'ha' (품사 정보가 없으면 정확도가 떨어질 수 있음)
print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in words])

print('표제어 추출 후 :', [lemmatizer.lemmatize(word, pos='v') for word in words])

print("="*200)

# 품사 정보(POS Tag)를 명시적으로 전달하는 경우의 예시
txts = ["lives", "dies", "watched", "has", "starting"]

for txt in txts:
  # lemmatize(단어, 품사) 형식으로 사용
  # 'v'는 동사(Verb)를 의미합니다. 
  # 동사임을 알려주면 'dies'는 'die'로, 'watched'는 'watch'로 정확하게 기본형을 찾아갑니다.
  print(f"{txt} -> {lemmatizer.lemmatize(txt, 'v')}")

print("="*200)
