from src import step01, step02, step03, step04, step05, step06, step07, step08, step09, step10, step11, step12, step13, step14
import asyncio

stream_modes = [
  "updates", # 각 상태 업데이트 시점마다 전체 상태를 반환
  "values", # 각 상태 업데이트 시점마다 변경된 필드와 값만 반환
  "messages", # 각 상태 업데이트 시점마다 메시지 필드의 변경된 메시지만 반환
]

def main():
  print("Hello from app!")
  # print("Step 1 실행: TypedDict 사용")
  # step01.run()
  # print("Step 2 실행: Pydantic 사용")
  # step02.run()
  # print("Step 3 실행: LangGraph 사용하여 그래프 이미지 저장")
  # step03.run()
  # print("Step 4 실행: 간단한 챗봇 노드(Node) 만들기")
  # step04.run()
  # print("Step 5 실행: 조건부 엣지 사용")
  # step05.run()
  # print("Step 6 실행: 대화메시지 상태 업데이트")
  # step06.run()
  # print("Step 7 실행: 대화메시지 상태 누적 업데이트")
  # step07.run()
  # print("Step 8 실행: 비동기 대화메시지 상태 누적 업데이트")
  # asyncio.run(step08.run())
  # print("Step 9 실행: 스트리밍 모드로 대화메시지 상태 누적 업데이트")
  # step09.run(stream_modes[0])
  # step09.run(stream_modes[1])
  step09.run(stream_modes[2])
  # print("Step 10 실행: 노드와 엣지 연결하기")
  # step10.run()
  # print("Step 11 실행: 한번에 노드 연결하기")
  # step11.run()
  # print("Step 12 실행: 병렬 그래프 사용")
  # step12.run()
  # print("Step 13 실행: Fan-in과 Fan-out 패턴 구현하기")
  # step13.run()
  # print("Step 14 실행: 조건에 따른 반복 처리 구현하기")
  # step14.run()

if __name__ == "__main__":
  main()
