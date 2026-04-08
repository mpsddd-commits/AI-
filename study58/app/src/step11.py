from src.save_image import save_graph_image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START

# 1. 상태(State) 정의
# 그래프 내에서 공유할 데이터 구조를 정의합니다.
class State(TypedDict):
  value_1: str
  value_2: int

# 2. 개별 노드(Node) 함수 정의
# 각 함수는 상태를 입력받아 변경하거나 추가할 부분만 딕셔너리로 반환합니다.
def step_1(state: State):
  """입력받은 value_1을 확인하고 그대로 반환"""
  return {"value_1": state["value_1"]}

def step_2(state: State):
  """기존 value_1 뒤에 ' b'를 붙여 업데이트"""
  current_value_1 = state["value_1"]
  return {"value_1": f"{current_value_1} b"}

def step_3(state: State):
  """새로운 필드 value_2에 10을 할당"""
  return {"value_2": 10}

def run():
  try:
    # 3. 그래프 구성 (add_sequence 사용)
    # add_sequence([A, B, C])는 A -> B -> C 순서로 노드 추가와 엣지 연결을 한 번에 수행합니다.
    # 하나씩 add_node, add_edge를 호출할 필요가 없어 코드가 매우 간결해집니다.
    graph_builder = StateGraph(State).add_sequence([step_1, step_2, step_3])
    
    # 4. 시작점(Entry Point) 설정
    # 시퀀스의 첫 번째 노드인 "step_1"으로 진입하도록 설정합니다.
    graph_builder.add_edge(START, "step_1")
    
    # 5. 그래프 컴파일
    # 설정된 시퀀스를 바탕으로 실행 가능한 그래프 객체 생성
    graph = graph_builder.compile()
    
    # 그래프 구조 시각화 (이미지 저장)
    save_graph_image(graph)
    
    # 6. 그래프 실행 (Invoke)
    # 초기 입력값으로 value_1에 "test"를 전달합니다.
    # 흐름: START -> step_1 ("test") -> step_2 ("test b") -> step_3 (value_2: 10 추가)
    result = graph.invoke({"value_1": "test"})
    
    # 7. 최종 결과 출력
    # 최종적으로 모든 가공이 완료된 상태(State)를 확인합니다.
    print("\n최종 결과:", result)
      
  except Exception as e:
    # 설정 오류나 실행 중 발생하는 예외 처리
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
