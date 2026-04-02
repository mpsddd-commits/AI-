from src import step1, step2, step3, step4, step5

def main():
	print("1. Step 1: 훈련 데이터 준비")
	step1.run() # 한줄로 만들기
	print("2. Step 2: 토크나이저 테스트")
	step2.run()
	print("3. Step 3: 데이터 로더 정의")
	datasets = step3.run(10000) # 이거 돌리면 x에 값이 나오고 y에는 바로 뒤의 토큰이 나온다
	for i, dataset in enumerate(datasets):
		print(f"{i} 모델 정의 및 훈련")
		step4.run(dataset)
	print("4. Step 5: 모델 테스트")
	step5.run()

if __name__ == "__main__":
  main()
