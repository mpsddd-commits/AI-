from src.save_image import save_graph_image
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# 1. 상태(State) 정의
class State(TypedDict):
  # add_messages: 노드에서 반환하는 메시지를 기존 리스트에 누적(Append)
  messages: Annotated[list[AnyMessage], add_messages]
  extra_field: int

# 2. 노드(Node) 정의
def node(state: State):
  """간단한 인사를 반환하는 노드"""
  new_message = AIMessage("안녕하세요! 무엇을 도와드릴까요?")
  return {"messages": new_message, "extra_field": 10}

# 3. 스트림 모드별 출력 함수들
def updates_print(graph, input_message):
  """
  [updates 모드]
  - 특정 노드에서 발생한 '변경 사항(Delta)'만 출력합니다.
  - 출력 형태: {'노드이름': {'업데이트된 필드': 값}}
  """
  for chunk in graph.stream({"messages": [input_message]}, stream_mode="updates"):
    print(chunk) # 전체 청크 출력
    for node_name, value in chunk.items():
      if node_name:
        print(f"현재 실행된 노드: {node_name}")
      if "messages" in value:
        # 해당 노드에서 새로 추가된 메시지의 내용만 출력
        print(f"추가된 메시지: {value['messages'].content}")

def values_print(graph, input_message):
  """
  [values 모드]
  - 노드가 실행될 때마다 그래프의 '전체 상태(Full State)'를 출력합니다.
  - 출력 형태: {'messages': [...], 'extra_field': 10}
  """
  for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    print(chunk) # 현재 시점의 전체 상태 출력
    for state_key, state_value in chunk.items():
      if state_key == "messages":
        # 전체 대화 내역 중 가장 마지막 메시지(방금 추가된 것) 출력
        state_value[-1].pretty_print()

def messages_print(graph, input_message):
  """
  [messages 모드]
  - LLM 응답 토큰이나 메시지 단위의 이벤트를 출력합니다.
  - 출력 형태: (Message 객체, Metadata) 튜플 형태
  """
  for chunk_msg, metadata in graph.stream({"messages": [input_message]}, stream_mode="messages"):
    print(f"메시지 객체: {chunk_msg}")
    print(f"내용: {chunk_msg.content}")
    print(f"메타데이터: {metadata}")
    # 어느 노드에서 이 메시지가 생성되었는지 확인 가능
    print(f"발생 노드: {metadata['langgraph_node']}")

# 4. 실행 메인 함수
def run(stream_mode: str = "updates"):
  try:
    # 그래프 빌드 및 컴파일
    graph_builder = StateGraph(State)
    graph_builder.add_node("node", node)
    graph_builder.add_edge(START, "node")
    graph_builder.add_edge("node", END)
    graph = graph_builder.compile()
    
    # 구조 시각화 저장
    save_graph_image(graph)
    
    input_message = {"role": "user", "content": "안녕하세요."}
    print(f"\n--- 현재 테스트 중인 stream_mode: {stream_mode} ---")
    
    # 선택된 모드에 따라 해당 출력 함수 호출
    if "values" == stream_mode:
      values_print(graph, input_message)
    elif "messages" == stream_mode:
      messages_print(graph, input_message)
    else:
      updates_print(graph, input_message)
          
  except Exception as e:
    print(f"오류 발생: {e}")

if __name__ == "__main__":
  # 실행 시 원하는 모드를 인자로 전달하여 테스트 가능 ("updates", "values", "messages")
  run("updates")
