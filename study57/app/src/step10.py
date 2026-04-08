from settings import settings
from src.save_image import save_graph_image
# operator: 파이썬의 표준 연산자 함수 모듈
# LangGraph에서 'Reducer' 기능을 구현할 때 사용합니다.
# 특히 'operator.add'는 기존 상태값에 새로운 값을 더하거나(숫자), 
# 리스트에 항목을 추가(Append)할 때 매우 유용합니다.
import operator
# typing.Annotated: 변수 타입에 추가적인 메타데이터를 부여합니다.
# LangGraph에서는 특정 상태(State) 필드가 어떻게 업데이트되어야 하는지
# '지침(Reducer)'을 내릴 때 사용합니다. (예: 단순히 덮어쓸지, 합칠지 결정)
from typing import Annotated
# typing_extensions.TypedDict: 딕셔너리의 키와 값의 타입을 고정합니다.
# 그래프 전체에서 공유되는 'State(상태)' 객체의 형식을 정의하는 도구입니다.
# 어떤 데이터가 오고 가는지 구조를 명확히 하여 버그를 방지합니다.
from typing_extensions import TypedDict
# langgraph.graph: 그래프 구조를 설계하는 핵심 클래스들
# - StateGraph: 노드와 엣지를 추가하여 실제 워크플로우를 설계하는 설계도 역할을 합니다.
# - START: 그래프가 시작되는 지점을 나타내는 특수 상수입니다.
# - END: 그래프의 실행이 완전히 종료되는 지점을 나타내는 특수 상수입니다.
from langgraph.graph import StateGraph, START, END

# 1. 그래프 상태(State) 정의
# 노드 간에 전달될 데이터의 구조를 정의합니다.
class State(TypedDict):
  counter: int
  # alphabet은 Annotated를 통해 'reducer' 함수를 지정했습니다.
  # operator.add가 지정되면, 노드에서 반환된 리스트를 기존 리스트에 '합칩니다(Append)'.
  # 이를 통해 대화 기록이나 데이터를 누적할 수 있습니다.
  alphabet: Annotated[list[str], operator.add]

# 2. 노드(Node) 함수 정의
# 특정 단계에서 실행될 비즈니스 로직입니다.
def node_a(state: State):
  print("--- 노드 A 실행 중 ---")
  # 현재 상태값을 읽어와서(get) 1을 더하고, 리스트를 업데이트한 결과를 반환합니다.
  # 반환된 값은 그래프의 현재 상태에 자동으로 병합(Merge)됩니다.
  return {
    "counter": state.get("counter", 0) + 1,
    "alphabet": ["Hello"]
  }

def run():
  try:
    # 그래프 빌더 초기화 (State 구조 기반)
    graph_builder = StateGraph(State)
    
    # 노드 배치 및 연결
    graph_builder.add_node("chatbot", node_a)      # 'chatbot'이라는 이름의 작업 노드 추가
    graph_builder.add_edge(START, 'chatbot')      # 시작점(START)에서 chatbot으로 연결
    graph_builder.add_edge('chatbot', END)        # chatbot 작업 완료 후 종료(END)로 연결
    
    # 그래프 컴파일 (실행 가능한 객체 생성)
    graph = graph_builder.compile()
    
    # 구조 시각화 저장
    save_graph_image(graph)
    
    # 초기 상태값 설정 후 그래프 실행
    initial_state = {"counter": 0, "alphabet": []}
    state = initial_state
    
    # 루프 실행: 그래프를 총 3번 연속으로 호출합니다.
    for i in range(3):
      print(f"\n--- {i+1}번째 실행 시작 ---")
      # 이전 실행의 결과물(state)을 다음 실행의 입력으로 넣습니다.
      state = graph.invoke(state)
      # 실행마다 alphabet 리스트에 "Hello"가 누적되는 것을 볼 수 있습니다.
      print(f"현재 상태: {state}")
    
    # 최종 결과 출력
    print("최종 상태 결과:", state)
  except Exception as e:
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
