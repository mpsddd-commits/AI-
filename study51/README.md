## 텍스트 전처리(Text preprocessing)

1. 토큰화(Tokenization) : 자연어 처리(NLP)의 가장 기초적인 단계
2. 표제어 추출(Lemmatization) : 단어들이 다른 형태를 가지더라도 그 뿌리 단어(기본 사전형 단어)를 찾아가서 단어의 개수를 줄이는 중요한 작업
3. 어간 추출(Stemming) : 표제어 추출(Lemmatization)과 달리, 정해진 규칙에 따라 단어의 뒷부분을 물리적으로 잘라내는 방식
4. 불용어(Stopword) : 전처리 과정의 필수 단계인 불용어(Stopwords) 제거
5. 정수 인코딩(Integer Encoding) : 텍스트 데이터를 숫자 형태로 변환 과정
6. 패딩(Padding) : 서로 다른 길이의 문장들을 동일한 길이로 맞춰주는 작업
7. 원-핫 인코딩(One-Hot Encoding) : 단어를 컴퓨터가 이해할 수 있는 벡터로 변환하는 가장 기본적인 방법
8. 데이터 분리(Splitting Data) : **훈련 데이터(Train set)**와 성능 평가를 위한 **테스트 데이터(Test set)**로 분리하는 방법
