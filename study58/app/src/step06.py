from src.save_image import save_graph_image
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph

# 1. 그래프의 상태(State) 정의
# 그래프 내에서 노드 간에 전달될 데이터의 구조를 정의합니다.
class State(TypedDict):
  # messages: 채팅 내역을 담는 리스트 (HumanMessage, AIMessage 등)
  messages: list[AnyMessage]
  # extra_field: 정수형 데이터를 담는 추가 필드 (상태 관리 예시)
  extra_field: int

# 2. 노드(Node) 함수 정의
# 그래프의 각 단계에서 실행될 로직을 작성합니다.
def node(state: State):
  """
  현재 상태(state)를 입력받아 처리한 후, 업데이트된 상태를 반환합니다.
  """
  # 현재 상태에서 메시지 목록을 가져옴
  messages = state["messages"]
  
  # AI의 응답 메시지 생성
  new_message = AIMessage("안녕하세요! 무엇을 도와드릴까요?")
  
  # 업데이트할 필드를 딕셔너리 형태로 반환
  # LangGraph는 반환된 값을 기존 상태에 병합(Merge)하거나 덮어씁니다.
  return {
    "messages": messages + [new_message], 
    "extra_field": 10
  }

def run():
  try:
    # 3. 그래프 빌더 초기화
    # 정의한 State 구조를 사용하는 StateGraph 객체를 생성합니다.
    graph_builder = StateGraph(State)
    
    # 4. 노드 추가 및 경로 설정
    # "node"라는 이름으로 위에서 정의한 node 함수를 등록합니다.
    graph_builder.add_node("node", node)
    
    # 그래프가 시작될 때 가장 먼저 실행될 노드(Entry Point)를 지정합니다.
    graph_builder.set_entry_point("node")
    
    # 5. 그래프 컴파일
    # 설정을 마친 빌더를 실행 가능한 'graph' 객체로 변환합니다.
    graph = graph_builder.compile()
    
    # 6. 그래프 시각화 (선택 사항)
    # 그래프 구조를 이미지 파일로 저장하는 사용자 정의 함수 호출
    save_graph_image(graph)
    
    # 7. 그래프 실행 (Invoke)
    # 초기 입력값(HumanMessage)을 넣어 그래프를 시작합니다.
    initial_input = {"messages": [HumanMessage("안녕하세요.")]}
    result = graph.invoke(initial_input)
    
    # 8. 결과 출력
    print("--- 실행 결과 ---")
    for message in result["messages"]:
      # 메시지 타입(Human/AI)과 내용을 보기 좋게 출력
      message.pretty_print()
        
    # 전체 상태 정보 확인
    print("\n최종 결과 메시지 객체:", result["messages"])
    print("추가 필드 값(extra_field):", result["extra_field"])

  except Exception as e:
    # 실행 중 발생할 수 있는 에러 처리
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
