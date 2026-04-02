from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt

# 1. NLTK가 제공하는 영어 불용어 확인
# NLTK는 미리 정의된 언어별 불용어 리스트를 가지고 있습니다.
stop_words_list = stopwords.words('english')
print('불용어 개수 :', len(stop_words_list)) # 약 198개의 불용어가 정의되어 있음
print('불용어 10개 출력 :', stop_words_list[:10])

print("="*200)

# 2. 영어 문장에서 불용어 제거하기
example = "Family is not an important thing. It's everything."
# 검색 속도를 높이기 위해 set(집합) 자료형으로 변환
stop_words = set(stopwords.words('english')) 

# 문장을 단어 단위로 토큰화
word_tokens = word_tokenize(example)

# 불용어 리스트에 포함되지 않은 단어만 추출
result = []
for word in word_tokens: 
    if word not in stop_words: 
        result.append(word) 

print('불용어 제거 전 :', word_tokens) 
# 'is', 'an' 같은 단어들이 제거된 결과 출력
print('불용어 제거 후 :', result)

print("="*200)

# 3. 한국어 불용어 제거하기
# 한국어는 조사, 접속사 등이 다양하여 분석가가 직접 불용어 리스트를 정의하는 경우가 많습니다.
okt = Okt()

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
# 분석 목적에 따라 제외하고 싶은 단어들을 나열
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

# 문자열을 공백 기준으로 나눠서 set으로 저장
stop_words = set(stop_words.split(' '))

# Okt 형태소 분석기를 사용하여 형태소 단위로 분리
word_tokens = okt.morphs(example)

# 리스트 컴프리헨션을 사용하여 불용어에 포함되지 않은 토큰만 필터링
result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 :', word_tokens) 
# 직접 정의한 stop_words에 해당하는 단어들이 필터링된 결과 출력
print('불용어 제거 후 :', result)
