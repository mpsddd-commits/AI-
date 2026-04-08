from src.save_image import save_graph_image
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 상태(State) 정의
class State(TypedDict):
  # [중요] Annotated와 operator.add를 사용하면 해당 필드는 'Reducer'가 됩니다.
  # 여러 노드에서 "aggregate" 값을 반환하면, 기존 값을 덮어쓰지 않고 
  # 리스트 더하기(operator.add)를 통해 결과를 모두 합칩니다.
  aggregate: Annotated[list, operator.add]

# 2. 노드(Node) 함수 정의
# 각 노드는 리스트에 자신의 이름을 담아 반환합니다.
def a(state: State):
  print(f'노드 A 실행 중: 현재 상태 {state["aggregate"]}')
  return {"aggregate": ["A"]}

def b(state: State):
  # 노드 A 다음에 실행됩니다.
  print(f'노드 B 실행 중: 현재 상태 {state["aggregate"]}')
  return {"aggregate": ["B"]}

def c(state: State):
  # 노드 A 다음에 실행됩니다. (노드 B와 병렬로 실행될 수 있음)
  print(f'노드 C 실행 중: 현재 상태 {state["aggregate"]}')
  return {"aggregate": ["C"]}

def d(state: State):
  # 노드 B와 C가 모두 완료된 후 실행됩니다. (Fan-in)
  print(f'노드 D 실행 중: 현재 상태 {state["aggregate"]}')
  return {"aggregate": ["D"]}

def run():
  try:
    # 3. 그래프 빌더 초기화
    graph_builder = StateGraph(State)
    
    # 4. 노드 추가
    graph_builder.add_node(a)
    graph_builder.add_node(b)
    graph_builder.add_node(c)
    graph_builder.add_node(d)
    
    # 5. 엣지(Edge) 연결 - 다이아몬드 구조 (Fan-out & Fan-in)
    # 시작 -> A
    graph_builder.add_edge(START, "a")
    
    # [Fan-out] A가 끝나면 B와 C가 동시에 활성화됩니다.
    graph_builder.add_edge("a", "b")
    graph_builder.add_edge("a", "c")
    
    # [Fan-in] B와 C가 모두 끝나면 D로 모입니다.
    graph_builder.add_edge("b", "d")
    graph_builder.add_edge("c", "d")
    
    # D가 끝나면 종료
    graph_builder.add_edge("d", END)
    
    # 6. 그래프 컴파일
    graph = graph_builder.compile()
    
    # 그래프 구조 시각화 (다이아몬드 형태의 그래프가 생성됨)
    save_graph_image(graph)
    
    # 7. 그래프 실행 (Invoke)
    # 빈 리스트로 시작합니다.
    # 예상 흐름: [] -> [A] -> [A, B, C] (혹은 A, C, B) -> [A, B, C, D]
    result = graph.invoke({"aggregate": []})
    
    # 8. 최종 결과 출력
    print("\n최종 결과:", result)
      
  except Exception as e:
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
