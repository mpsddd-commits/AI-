from src.save_image import save_graph_image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 상태(State) 구조 정의
# 그래프 내의 모든 노드가 공유하고 업데이트할 데이터 주머니입니다.
class State(TypedDict):
  input: str   # 사용자의 입력값
  output: str  # 최종 처리 결과물

# 2. 노드(Node) 함수 정의
# 각 단계에서 실제로 수행될 비즈니스 로직입니다.
def node_a(state: State):
  """첫 번째 노드: 입력을 받고 다음 단계를 결정하기 전의 전처리 단계"""
  print("--- 노드 A 실행 ---")
  return state  # 별도의 수정 없이 현재 상태를 그대로 전달

def node_b(state: State):
  """라우팅 성공 시 실행되는 노드"""
  print("--- 노드 B 실행 (Route 성공) ---")
  # 상태의 output 필드에 결과를 담아 반환
  return {"output": "Route B로 이동했습니다."}

def node_c(state: State):
  """라우팅 실패 시 실행되는 노드"""
  print("--- 노드 C 실행 (Route 실패) ---")
  return {"output": "Route C로 이동했습니다."}

# 3. 조건부 라우팅 함수 (Router)
# 이 함수의 반환값(Key)에 따라 다음에 실행될 노드가 결정됩니다.
def routing_function(state: State):
  # 입력값이 "isroute"인 경우와 그렇지 않은 경우를 판별
  if state.get("input") == "isroute":
    return "go_to_b"  # 조건 충족 시 반환할 키값
  return "go_to_c"      # 조건 미충족 시 반환할 키값

def run():
  try:
    # 4. 그래프 빌더 초기화
    # 정의한 State 구조를 기반으로 그래프의 뼈대를 만듭니다.
    graph_builder = StateGraph(State)

    # 5. 노드 등록
    # 사용할 모든 함수를 그래프의 노드로 등록합니다.
    graph_builder.add_node("node_a", node_a)
    graph_builder.add_node("node_b", node_b)
    graph_builder.add_node("node_c", node_c)

    # 6. 엣지(흐름) 연결
    # 시작 지점(START)에서 바로 node_a로 이동합니다.
    graph_builder.add_edge(START, "node_a")

    # [핵심] 조건부 엣지 설정
    # node_a 작업이 끝난 후, routing_function의 결과에 따라 갈림길을 정합니다.
    graph_builder.add_conditional_edges(
      "node_a",           # 시작 노드
      routing_function,   # 판단 함수
      {
        "go_to_b": "node_b", # 함수 결과가 "go_to_b"면 node_b로 이동
        "go_to_c": "node_c"  # 함수 결과가 "go_to_c"면 node_c로 이동
      }
    )

    # 각 분기 노드가 끝나면 그래프를 종료(END)합니다.
    graph_builder.add_edge("node_b", END)
    graph_builder.add_edge("node_c", END)

    # 7. 그래프 컴파일
    # 모든 설정이 끝난 설계도를 실행 가능한 객체로 변환합니다.
    graph = graph_builder.compile()

    # 구조 시각화 이미지 저장 (커스텀 함수 호출)
    save_graph_image(graph)

    # 8. 실제 실행 테스트
    print("\n[테스트 1: isroute 입력]")
    # invoke 시 전달한 딕셔너리가 초기 State가 됩니다.
    graph.invoke({"input": "isroute"})

    print("\n[테스트 2: empty 입력]")
    graph.invoke({"input": ""})
  except Exception as e:
    # 실행 중 발생하는 모든 오류를 잡아내어 출력합니다.
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
