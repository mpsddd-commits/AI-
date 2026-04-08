from src import step1, step2, step3, step4, step5, step6, step7, step8, step9

def main():
  # print("1. Step 1: 훈련 데이터 준비")
  # step1.run()
  # print("2. Step 2: 토크나이저 테스트")
  # step2.run()
  # print("3. Step 3: 데이터 로더 정의")
  # datasets = step3.run(10000)
  # for i, dataset in enumerate(datasets):
  #   print(f"{i} 모델 정의 및 훈련")
  # #   step4.run(dataset)
  #   step6.run(dataset, "20260403_1049/003.pth")
	# print("4. Step 5: 모델 테스트")
	step5.run("20260403_1558")
  # print("5. Step 7: 모델 변환")
  # step7.run("003.pth", 1)
  # print("6. Step 8: 모델 테스트")
  # step8.run("20260403_1558/001.pth")
  # step9.run("20260403_1046")

if __name__ == "__main__":
  main()
