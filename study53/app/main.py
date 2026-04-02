from konlpy.tag import Okt

# Okt 형태소 분석기 객체를 생성합니다.
okt = Okt()

def build_wordIndex(token):
  """
  토큰 리스트를 입력받아 각 단어에 고유한 정수 인덱스를 부여하는 
  단어 사전(Word Index)을 생성하는 함수입니다.
  """
  wordIndex = {}
  for voca in token:
    # 단어가 아직 사전에 없다면, 현재 사전의 길이를 인덱스로 사용하여 추가합니다.
    # (예: 첫 번째 단어는 index 0, 두 번째는 index 1...)
    if voca not in wordIndex.keys():
      wordIndex[voca] = len(wordIndex)
  return wordIndex

def one_hot_encoding(word, word2index):
  """
  특정 단어를 원-핫 벡터로 변환하는 함수입니다.
  선택된 단어의 인덱스 위치만 1이고, 나머지는 모두 0인 리스트를 만듭니다.
  """
  # 1. 전체 단어 집합의 크기만큼 0으로 채워진 리스트를 생성합니다.
  one_hot_vector = [0] * (len(word2index))
  
  # 2. 해당 단어의 고유 인덱스를 사전에서 찾아옵니다.
  index = word2index[word]
  
  # 3. 해당 인덱스 위치의 값을 1로 변경합니다.
  one_hot_vector[index] = 1
  
  return one_hot_vector

def main():
  # 1. 형태소 분석을 통해 문장을 토큰화합니다.
  # 결과: ['나', '는', '자연어', '처리', '를', '배운다']
  token = okt.morphs("나는 자연어 처리를 배운다")
  print(f"토큰화 : {token}")

  # 2. 토큰화된 리스트를 바탕으로 중복 없는 단어 집합(사전)을 만듭니다.
  # 결과: {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}
  wordIndex = build_wordIndex(token)
  print(f"단어 집합 : {wordIndex}")

  # 3. '자연어'라는 단어를 원-핫 벡터로 변환합니다.
  # '자연어'는 인덱스 2번이므로 리스트의 3번째 값이 1이 됩니다.
  result = one_hot_encoding("자연어", wordIndex)
  print(f"'자연어'라는 단어를 원핫벡터 : {result}")

  # 4. 토큰화된 리스트를 바탕으로 원-핫 벡터로 변환합니다.
  vectorList = []
  for voca in token:
    result = one_hot_encoding(voca, wordIndex)
    vectorList.append(result)
  print(f"원핫벡터 리스트 : {vectorList}")

# 스크립트가 직접 실행될 경우 main 함수를 호출합니다.
if __name__ == "__main__":
  main()