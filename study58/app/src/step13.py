from src.save_image import save_graph_image
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 그래프 상태(State) 정의
class State(TypedDict):
  # Annotated와 operator.add를 사용하여 여러 노드의 리스트 출력을 하나로 합칩니다 (Reducer).
  aggregate: Annotated[list, operator.add]
  # 어떤 경로를 탈지 결정하는 조건 값
  which: str

# 2. 노드(Node) 함수 정의
def nodeA(state: State):
  print(f'--- Node A: 현재 aggregate = {state.get("aggregate", [])} ---')
  return {"aggregate": ["A"]}

def nodeB(state: State):
  print(f'--- Node B 실행 (B 추가) ---')
  return {"aggregate": ["B"]}

def nodeC(state: State):
  print(f'--- Node C 실행 (C 추가) ---')
  return {"aggregate": ["C"]}

def nodeD(state: State):
  print(f'--- Node D 실행 (D 추가) ---')
  return {"aggregate": ["D"]}

def nodeE(state: State):
  """
  Fan-in 지점: 이전의 병렬 노드들이 모두 완료될 때까지 기다린 후 실행됩니다.
  """
  print(f'--- Node E (Fan-in 지점): 현재 aggregate = {state["aggregate"]} ---')
  return {"aggregate": ["E"]}

# 3. 조건부 라우팅 함수 정의
def route_bc_or_cd(state: State) -> Sequence[str]:
  """
  상태의 'which' 값에 따라 다음에 실행할 노드 리스트를 반환합니다.
  리스트를 반환하면 해당 노드들이 '병렬'로 실행됩니다.
  """
  if state["which"] == "cd":
    # which가 "cd"라면 Node C와 Node D를 실행
    return ["nodeC", "nodeD"]
  # 기본값으로 Node B와 Node C를 실행
  return ["nodeB", "nodeC"]

def run():
  try:
    # 4. 그래프 빌더 생성
    graph_builder = StateGraph(State)
    
    # 노드 등록
    graph_builder.add_node(nodeA)
    graph_builder.add_node(nodeB)
    graph_builder.add_node(nodeC)
    graph_builder.add_node(nodeD)
    graph_builder.add_node(nodeE)

    # 5. 엣지 연결 (흐름 제어)
    # START -> Node A
    graph_builder.add_edge(START, "nodeA")

    # [핵심] 조건부 엣지 설정
    # Node A가 끝난 후 route_bc_or_cd 함수의 결과에 따라 분기합니다.
    # 세 번째 인자는 이 라우터가 갈 수 있는 모든 후보 노드의 이름 리스트입니다.
    graph_builder.add_conditional_edges(
      "nodeA",
      route_bc_or_cd,
      ["nodeB", "nodeC", "nodeD"]
    )

    # 각 병렬 후보 노드에서 최종 노드(Node E)로 모이도록 설정 (Fan-in)
    graph_builder.add_edge("nodeB", "nodeE")
    graph_builder.add_edge("nodeC", "nodeE")
    graph_builder.add_edge("nodeD", "nodeE")
    
    # Node E가 끝나면 종료
    graph_builder.add_edge("nodeE", END)

    # 6. 그래프 컴파일 및 시각화
    graph = graph_builder.compile()
    save_graph_image(graph)

    # 7. 그래프 실행
    # which를 "bc"로 설정했으므로 Node A -> (Node B, Node C 병렬 실행) -> Node E 순서로 진행됩니다.
    result = graph.invoke({"aggregate": [], "which": "bc"})
    
    # 8. 결과 출력
    # 예상 결과: ["A", "B", "C", "E"] (B와 C의 순서는 바뀔 수 있음)
    print("\n최종 결과:", result["aggregate"])
      
  except Exception as e:
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
