from src.save_image import save_graph_image
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

# 1. 상태(State) 정의
class State(TypedDict):
  # operator.add를 사용하여 노드들이 반환하는 리스트를 기존 상태에 계속 합칩니다.
  aggregate: Annotated[list, operator.add]

# 2. 노드(Node) 함수 정의
def a(state: State):
  """노드 A: 현재 리스트 상태를 출력하고 'A'를 추가함"""
  print(f'Node A 처리 중 현재 상태값 : {state["aggregate"]}')
  return {"aggregate": ["A"]}

def b(state: State):
  """노드 B: 현재 리스트 상태를 출력하고 'B'를 추가함"""
  print(f'Node B 처리 중 현재 상태값 : {state["aggregate"]}')
  return {"aggregate": ["B"]}

# 3. 조건부 라우팅(Router) 함수 정의
def route(state: State):
  """
  현재 상태의 길이를 체크하여 다음 행방을 결정합니다.
  - 길이가 7보다 작으면: Node B로 이동 (루프 지속)
  - 길이가 7 이상이면: END로 이동 (그래프 종료)
  """
  print(f'라우팅 결정 중 현재 상태값 : {state["aggregate"]}')
  if len(state["aggregate"]) < 7:
    return "b"
  else:
    return END

def run():
  try:
    # 4. 그래프 구축
    graph_builder = StateGraph(State)
    
    # 노드 등록
    graph_builder.add_node(a)
    graph_builder.add_node(b)
    
    # 5. 엣지 및 순환(Loop) 설정
    # 시작점 -> A
    graph_builder.add_edge(START, "a")
    
    # [핵심] A 노드 이후의 행방을 route 함수에 맡김 (A -> B 혹은 A -> END)
    graph_builder.add_conditional_edges("a", route)
    
    # [순환] B 노드가 끝나면 다시 A 노드로 돌아감 (B -> A)
    graph_builder.add_edge("b", "a")
    
    # 그래프 컴파일
    graph = graph_builder.compile()
    
    # 그래프 구조 시각화 (A와 B가 서로 화살표를 주고받는 순환 구조 확인 가능)
    save_graph_image(graph)
    
    # 6. 그래프 실행 및 재귀 제한 설정
    # config 매개변수의 'recursion_limit'은 그래프 내에서 노드가 실행되는 최대 횟수를 제한합니다.
    # 이 코드의 의도대로라면 ["A", "B", "A", "B", ...] 순서로 총 7개 이상 쌓여야 종료되지만,
    # 제한을 4로 두었기 때문에 4번째 노드 실행 시점에서 에러가 발생합니다.
    graph.invoke({"aggregate": []}, config={"recursion_limit": 10})

  except GraphRecursionError as e:
    # 설정한 재귀 제한(recursion_limit)을 초과했을 때 발생하는 에러 처리
    print(f"Recursion Error: {e}")
  except Exception as e:
    # 기타 발생할 수 있는 에러 처리
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
