from src.save_image import save_graph_image
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph

# 1. 그래프 상태(State) 정의
class State(TypedDict):
  # Annotated와 add_messages를 사용하면, 
  # 노드에서 새로운 메시지를 반환할 때 기존 리스트에 자동으로 추가(Append)됩니다.
  # 이전 코드처럼 messages = messages + [new_message]를 할 필요가 없습니다. 그리고 state가 관리를 하기 때문에 step6과는 다르게 id의 값을 출력한다. 그리고 고유의 값이 있어 추적하거나 사용하기 용이하므로 step6보다 더 유용하게 사용할 수 있다.
  messages: Annotated[list[AnyMessage], add_messages]
  
  # 일반 필드는 노드에서 반환하는 값으로 계속 덮어쓰기(Overwrite) 됩니다.
  extra_field: int

# 2. 노드(Node) 함수 정의
def node(state: State):
  """
  현재 상태를 입력받아 로직을 수행하고, 업데이트할 상태를 반환합니다.
  """
  # state["messages"]에는 지금까지 쌓인 대화 내역이 들어 있습니다.
  messages = state["messages"]
  
  # 새로운 응답 생성
  new_message = AIMessage("안녕하세요! 무엇을 도와드릴까요?")
  
  # [중요] add_messages 덕분에 리스트 전체가 아닌 '새로운 메시지'만 반환해도 
  # 기존 대화 내역 뒤에 자동으로 붙습니다.
  return {"messages": new_message, "extra_field": 10}

def run():
  try:
    # 3. 그래프 생성 및 설정
    # State 구조를 기반으로 그래프 빌더 초기화
    graph_builder = StateGraph(State)
    
    # "node"라는 이름의 작업 단계를 등록
    graph_builder.add_node("node", node)
    
    # 시작 지점을 "node"로 설정
    graph_builder.set_entry_point("node")
    
    # 4. 그래프 컴파일 (실행 가능한 형태로 변환)
    graph = graph_builder.compile()
    
    # 그래프 구조 시각화 (이미지 저장)
    save_graph_image(graph)
    
    # 5. 초기 입력값 설정
    # 딕셔너리 형태의 메시지도 LangGraph가 내부적으로 적절한 객체로 변환합니다.
    input_message = {"role": "user", "content": "안녕하세요."}
    
    # 6. 그래프 실행 (Invoke)
    # 초기 메시지를 넣고 실행하면, 정의된 노드들을 거쳐 최종 result가 반환됩니다.
    result = graph.invoke({"messages": [input_message]})
    
    # 7. 실행 결과 출력
    print("--- 대화 기록 출력 ---")
    for message in result["messages"]:
      # 메시지 타입과 내용을 정돈된 형태로 출력
      message.pretty_print()
        
    print("\n최종 결과(리스트 형태):", result["messages"])
      
  except Exception as e:
    # 코드 실행 중 에러가 발생할 경우 출력
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  run()
