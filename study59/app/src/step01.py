from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from settings import settings
from src.save_image import save_graph_image

# 1. 모델 설정: Ollama를 사용하여 로컬 또는 원격 LLM 연결
model_name: str = "gemma4:e4b"
llm = OllamaLLM(
  model=model_name, 
  base_url=settings.ollama_base_url,
  streaming=True  # 실시간 응답 생성을 위한 스트리밍 활성화
)

# 2. 상태(State) 정의: 그래프 내에서 노드 간에 전달될 데이터 구조
class State(TypedDict):
  # add_messages: 새로운 메시지가 들어오면 기존 리스트에 덮어쓰지 않고 추가(append)하는 리듀서
  messages: Annotated[list[BaseMessage], add_messages]

# 3. 노드 함수 정의: 실제 챗봇의 로직을 담당
def chatbot(state: State):
  """
  현재 상태의 메시지 기록을 LLM에 전달하고 응답을 받아 상태를 업데이트합니다.
  """
  # state["messages"]에는 지금까지의 대화 내역이 들어있음
  response = llm.invoke(state["messages"])
  # AI의 응답을 메시지 객체로 감싸서 반환 (add_messages에 의해 기존 리스트에 추가됨)
  return {"messages": [AIMessage(content=response)]}

# 4. 그래프 구축 및 설정 함수
def setup_graph():
  # 그래프 빌더 초기화 (우리가 정의한 State 구조 사용)
  graph_builder = StateGraph(State)
  
  # "chatbot"이라는 이름의 노드 추가
  graph_builder.add_node("chatbot", chatbot)
  
  # 흐름 정의: 시작(START) -> chatbot 노드 -> 종료(END)
  graph_builder.add_edge(START, "chatbot")
  graph_builder.add_edge("chatbot", END)
  
  # 체크포인터(MemorySaver): 대화 기록을 메모리에 저장하여 동일한 thread_id일 때 문맥 유지
  memory = MemorySaver()
  
  # 그래프 컴파일: 체크포인터를 포함하여 실행 가능한 형태의 그래프 생성
  graph = graph_builder.compile(checkpointer=memory)
  
  # (선택 사항) 그래프의 시각적 구조를 이미지로 저장
  # save_graph_image(graph)
  return graph

# 5. 스트리밍 업데이트 함수: 사용자 입력을 처리하고 응답을 출력
def stream_graph_updates(user_input: str, graph: StateGraph, config: dict):
  # 사용자 입력을 HumanMessage 형태로 변환하여 입력값 구성
  inputs = {"messages": [HumanMessage(content=user_input)]}
  
  print("Assistant: ", end="", flush=True)
  
  # graph.stream: 스트리밍 모드로 그래프 실행
  # stream_mode="messages"는 LLM의 응답 토큰이 생성될 때마다 이벤트를 발생시킴
  for msg, metadata in graph.stream(inputs, config=config, stream_mode="messages"):
    # 메시지의 내용(content)이 비어있지 않으면 화면에 출력
    if msg.content:
      print(msg.content, end="", flush=True)
  print()

# 6. 메인 실행 루프
def run():
  try:
    # 그래프 생성
    graph = setup_graph()
    
    # 설정: thread_id는 특정 대화 세션을 구분하는 고유값 (여기서는 "1"로 고정)
    config = {"configurable": {"thread_id": "1"}}
    
    print(f"{model_name} 모델 챗봇이 시작되었습니다. (종료: q, quit, exit)")
    
    while True:
      # 사용자로부터 입력 받기
      user_input = input("User: ")
      
      # 종료 조건 확인
      if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
      
      # 챗봇 실행 및 스트리밍 응답 출력
      stream_graph_updates(user_input, graph, config)
          
  except Exception as e:
      print(f"오류 발생: {e}")

if __name__ == '__main__':
  run()
