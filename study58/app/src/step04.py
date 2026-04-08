from src.save_image import save_graph_image
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add

# 1. 그래프 상태(State) 정의
class State(TypedDict):
  # Annotated와 add(operator.add)를 사용하면 이 필드는 '리듀서(Reducer)' 모드로 동작합니다.
  # 노드에서 새로운 리스트를 반환할 때, 기존 리스트를 덮어쓰지 않고 
  # [기존 리스트] + [새 리스트] 형태로 데이터를 계속 누적(Append)합니다.
  messages: Annotated[list[str], add]

# 2. 챗봇 노드(Node) 함수 정의
def chatbot(state: State):
  # state["messages"]에는 사용자가 보낸 이전 메시지들이 들어있습니다.
  answer = "안녕하세요! 무엇을 도와드릴까요?"
  print("Answer : ", answer)
  
  # 새로운 응답을 리스트 형태로 반환합니다.
  # 위에서 설정한 'add' 규칙에 의해 이 값은 기존 메시지 리스트 끝에 추가됩니다.
  return {"messages": [answer]}

def run():
  # 3. 그래프 구성
  # StateGraph(State)를 사용하여 우리가 정의한 상태 구조를 기반으로 설계도를 만듭니다.
  # (제시하신 코드의 StateGraph(dict)보다 State 클래스를 넣는 것이 더 안전합니다.)
  graph_builder = StateGraph(State)
  
  # 노드 추가: 'chatbot'이라는 이름의 단계를 등록합니다.
  graph_builder.add_node("chatbot", chatbot)
  
  # 엣지 연결: 시작(START) -> 챗봇 노드 -> 종료(END) 순서로 흐름을 지정합니다.
  graph_builder.add_edge(START, "chatbot")
  graph_builder.add_edge("chatbot", END)
  
  # 4. 그래프 컴파일 및 시각화
  # 설계를 확정하고 실행 가능한 객체로 만듭니다.
  graph = graph_builder.compile()
  
  # 그래프 구조를 PNG 파일로 저장합니다.
  save_graph_image(graph)
  
  # 5. 그래프 실행 (Invoke)
  # 초기 상태값으로 사용자의 첫 인사를 넣습니다.
  initial_state = {"messages": ["안녕!"]}
  
  # 그래프를 실행하면 [사용자 메시지] + [AI 응답]이 합쳐진 결과가 나옵니다.
  final_output = graph.invoke(initial_state)
  
  # 최종 결과 출력
  print("최종 상태 결과:", final_output)

if __name__ == "__main__":
  run()
