from src.save_image import save_graph_image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START

# 1. 그래프 상태(State) 정의
# 그래프 전체에서 공유되는 데이터의 스키마를 정의합니다.
class State(TypedDict):
  value_1: str  # 문자열 데이터
  value_2: int  # 정수형 데이터

# 2. 노드(Node) 함수들 정의
# 각 노드는 현재 상태(state)를 입력받아 변경된 부분만 딕셔너리로 반환합니다.
def step_1(state: State):
  """첫 번째 단계: 전달받은 value_1을 그대로 다시 반환 (상태 유지 확인)"""
  print("--- step_1 실행 ---")
  return {"value_1": state["value_1"]}

def step_2(state: State):
  """두 번째 단계: 기존 value_1 문자열 뒤에 ' b'를 추가"""
  print("--- step_2 실행 ---")
  current_value_1 = state["value_1"]
  return {"value_1": f"{current_value_1} b"}

def step_3(state: State):
  """세 번째 단계: value_2에 숫자 10을 할당"""
  print("--- step_3 실행 ---")
  # value_1은 건드리지 않고 value_2만 업데이트합니다.
  return {"value_2": 10}

def run():
  try:
    # 3. 그래프 빌더 초기화
    graph_builder = StateGraph(State)
    
    # 4. 노드 추가 (Add Nodes)
    # 함수 자체를 전달하면 함수의 이름이 노드의 이름이 됩니다.
    graph_builder.add_node(step_1)
    graph_builder.add_node(step_2)
    graph_builder.add_node(step_3)

    # 5. 엣지 연결 (Add Edges - 흐름 정의)
    # START -> step_1 -> step_2 -> step_3 순서로 진행됩니다.
    graph_builder.add_edge(START, "step_1")
    graph_builder.add_edge("step_1", "step_2")
    graph_builder.add_edge("step_2", "step_3")

    # 6. 그래프 컴파일
    # 정의된 노드와 엣지를 바탕으로 실행 가능한 객체를 생성합니다.
    graph = graph_builder.compile()
    
    # 그래프 구조 시각화 (이미지 저장)
    save_graph_image(graph)

    # 7. 그래프 실행 (Invoke)
    # 초기값으로 value_1에 "test"를 넣어 시작합니다.
    # 실행 순서: START -> step_1 ("test") -> step_2 ("test b") -> step_3 (value_2: 10 추가)
    result = graph.invoke({"value_1": "test"})
    
    # 8. 최종 결과 출력
    # 모든 단계를 거친 후의 전체 상태(State)가 출력됩니다.
    print("\n최종 결과:", result)
      
  except Exception as e:
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
