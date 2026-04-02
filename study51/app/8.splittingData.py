import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. zip 함수를 이용한 분리
def step1():
  print("[1. zip 함수를 이용한 분리]")
  # zip은 동일한 개수를 가진 자료형들을 묶어주는 역할도 하지만, 
  # zip(*리스트) 형태로 넘기면 다시 해체(unzip)하는 역할도 합니다.
  X, Y = zip(['a', 1], ['b', 2], ['c', 3])
  print('X 데이터 :', X) # ('a', 'b', 'c')
  print('y 데이터 :', Y) # (1, 2, 3)

  # 리스트 안에 리스트가 있는 경우 언패킹(*)을 사용하여 분리
  sequences = [['a', 1], ['b', 2], ['c', 3]]
  X, Y = zip(*sequences)
  print('X 데이터 :', X)
  print('y 데이터 :', Y)
  print("="*100)

# 2. Pandas DataFrame을 이용한 분리
def step2():
  print("[2. Pandas DataFrame을 이용한 분리]")
  values = [
    ['당신에게 드리는 마지막 혜택!', 1],
    ['내일 뵐 수 있을지 확인 부탁드...', 0],
    ['도연씨. 잘 지내시죠? 오랜만입...', 0],
    ['(광고) AI로 주가를 예측할 수 있다!', 1]
  ]
  columns = ['메일 본문', '스팸 메일 유무']

  df = pd.DataFrame(values, columns=columns)
  print(df)

  # 열 이름(Column Name)을 지정하여 X(특징)와 Y(정답/레이블)를 분리
  X = df['메일 본문']
  Y = df['스팸 메일 유무']
  print(f'X 데이터 : {X.to_list()}')
  print(f'y 데이터 : {Y.to_list()}')
  print("="*100)

# 3. Numpy 슬라이싱을 이용한 분리
def step3():
  print("[3. Numpy 슬라이싱을 이용한 분리]")
  # 4x4 크기의 배열 생성 (0~15)
  np_array = np.arange(0,16).reshape((4,4))
  print(f'전체 데이터 :\n{np_array}')

  # 마지막 열(정답 데이터)만 제외하고 나머지를 X로, 마지막 열만 Y로 분리
  X = np_array[:, :3] # 모든 행, 인덱스 0~2번 열까지
  Y = np_array[:, 3]  # 모든 행, 마지막 3번 인덱스 열만
  print(f'X 데이터 :\n{X}')
  print(f'Y 데이터 :\n{Y}')
  print("="*100)

# 4. Scikit-learn의 train_test_split()을 이용한 분리
def step4():
  print("[4. Scikit-learn의 train_test_split()을 이용한 분리]")
  # 가장 많이 쓰이는 방법으로, 데이터를 무작위로 섞어서 훈련/테스트용으로 나눠줌
  # test_size: 테스트 데이터의 비율 (0.2면 20%)
  # random_state: 실행할 때마다 결과가 달라지지 않도록 고정하는 난수 시드값
  X, Y = np.arange(10).reshape((5, 2)), range(5)
  print(f'X 전체 데이터 :\n{X}')
  print(f'y 전체 데이터 :\n{list(Y)}')
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
  print(f'X 훈련 데이터 :\n{X_train}')
  print(f'X 테스트 데이터 :\n{X_test}')
  print("="*100)

# 5. random_state의 영향 확인하기
def step5():
  print("[5. random_state의 영향 확인하기]")
  X, Y = np.arange(10).reshape((5, 2)), range(5)

  # test_size 0.3 (30%) 비율로 분할
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
  print(f'Y 훈련 데이터 (random_state=1234) : {Y_train}')

  # random_state를 1로 변경했을 때 (결과가 달라짐)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
  print(f'Y 훈련 데이터 (random_state=1) : {Y_train}')

  # 다시 random_state를 1234로 고정했을 때 (이전 1234 실행 결과와 동일함)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
  print(f'Y 훈련 데이터 (random_state=1234) : {Y_train}')
  print("="*100)

# 6. 수동으로 데이터 분리하기 (슬라이싱 활용)
def step6():
  print("[6. 수동으로 데이터 분리하기 (슬라이싱 활용)]")
  # 무작위로 섞지 않고 데이터의 앞부분은 훈련, 뒷부분은 테스트로 나누는 방식
  X, Y = np.arange(0,24).reshape((12,2)), range(12)

  # 전체 데이터의 80% 지점을 계산
  num_of_train = int(len(X) * 0.8)
  num_of_test = int(len(X) - num_of_train)
  print('훈련 데이터의 크기 :', num_of_train)
  print('테스트 데이터의 크기 :', num_of_test)

  # 슬라이싱을 이용해 데이터를 앞뒤로 자름
  # 80% 지점부터 끝까지
  X_test = X[num_of_train:]
  Y_test = Y[num_of_train:]

  # 처음부터 80% 지점 전까지
  X_train = X[:num_of_train] 
  Y_train = Y[:num_of_train]

  print(f'X 훈련 데이터 :\n{X_train}')
  print(f'X 테스트 데이터 :\n{X_test}')

  print(f'Y 훈련 데이터 :\n{Y_train}')
  print(f'Y 테스트 데이터 :\n{Y_test}')

if __name__ == '__main__':
  # step1()
  # step2()
  # step3()
  # step4()
  # step5()
  step6()