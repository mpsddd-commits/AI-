from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

# 두 가지 서로 다른 어간 추출 알고리즘 객체 생성
# PorterStemmer: 가장 대중적이고 보수적인(덜 깎는) 규칙 기반 스테머
# LancasterStemmer: 포터보다 훨씬 강력하고 공격적인(더 많이 깎는) 규칙 기반 스테머
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

# 1. 문장 단위 어간 추출 실습
sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."

# 단어 토큰화 수행 (공백 및 구두점 기준 분리)
tokenized_sentence = word_tokenize(sentence)

print('어간 추출 전 :', tokenized_sentence)
# 포터 스테머를 적용하여 리스트 내 모든 단어의 어간을 추출
# 예: 'heights' -> 'height', 'soundings' -> 'sound' 등으로 변환됨
print('어간 추출 후 :', [porter_stemmer.stem(word) for word in tokenized_sentence])

print("="*200)

# 2. 두 스테머(Porter vs Lancaster)의 성능 비교 실습
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('어간 추출 전 :', words)

# 포터 스테머: 비교적 원형을 알아볼 수 있을 만큼만 변환
# 예: 'organization' -> 'organ' (일부 탈락)
print('포터 스테머의 어간 추출 후:', [porter_stemmer.stem(w) for w in words])

# 랭커스터 스테머: 매우 강력하게 단어를 자르기 때문에 원형을 알아보기 힘든 경우가 많음
# 예: 'organization' -> 'org' (대폭 탈락)
print('랭커스터 스테머의 어간 추출 후:', [lancaster_stemmer.stem(w) for w in words])

print("="*200)
