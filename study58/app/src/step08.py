from src.save_image import save_graph_image
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import asyncio

# 1. 그래프 상태(State) 정의
class State(TypedDict):
  # Annotated[list, add_messages]: 
  # 새로운 메시지가 들어오면 기존 리스트를 대체하지 않고 끝에 추가(append)하는 'Reducer' 기능을 수행합니다.
  messages: Annotated[list[AnyMessage], add_messages]
  
  # 일반 필드: 새로운 값이 들어오면 기존 값을 덮어씁니다(Overwrite).
  extra_field: int

# 2. 노드(Node) 정의: 실제 작업이 일어나는 함수
def node(state: State):
  """
  현재 상태를 입력받아 처리 후, 업데이트할 변경 사항만 딕셔너리로 반환합니다.
  """
  # state["messages"]에는 이전 대화 기록이 모두 포함되어 있습니다.
  messages = state["messages"]
  
  # AI 응답 생성
  new_message = AIMessage("안녕하세요! 무엇을 도와드릴까요?")
  
  # [중요] add_messages 덕분에 전체 리스트가 아닌 새 메시지 하나만 반환해도 합쳐집니다.
  return {"messages": new_message, "extra_field": 10}

# 3. 비동기 실행 함수 정의 (async)
async def run():
  try:
    # 그래프 빌더 초기화 (사용할 상태 구조 전달)
    graph_builder = StateGraph(State)
    
    # 노드 배치: "node"라는 이름으로 위에서 정의한 함수를 등록
    graph_builder.add_node("node", node)
    
    # 4. 엣지(Edge) 설정: 흐름의 시작과 끝을 명시
    # START 노드(입력 지점)에서 "node"로 연결
    graph_builder.add_edge(START, "node")
    
    # "node" 작업이 끝나면 END 노드(종료 지점)로 연결
    graph_builder.add_edge("node", END)
    
    # 5. 그래프 컴파일
    # 설정된 노드와 엣지를 바탕으로 실행 가능한 그래프 객체 생성
    graph = graph_builder.compile()
    
    # 그래프 구조 시각화 (파일 저장)
    save_graph_image(graph)
    
    # 6. 입력 데이터 준비
    # 딕셔너리 형태의 입력은 내부적으로 HumanMessage 등으로 자동 변환됩니다.
    input_message = {"role": "user", "content": "안녕하세요."}
    
    # 7. 비동기 실행 (ainvoke)
    # await를 사용하여 비동기적으로 그래프 실행 결과를 기다립니다. 비동기 특징으로 invoke부분에 a가붙는다.
    result = await graph.ainvoke({"messages": [input_message]})
    
    # 8. 결과 출력
    print("--- 전체 대화 내역 ---")
    for message in result["messages"]:
      message.pretty_print()
        
    print("\n최종 상태 값(extra_field):", result["extra_field"])

  except Exception as e:
      print(f"오류 발생: {e}")

if __name__ == "__main__":
  # 비동기 함수 실행을 위한 표준 방식
  asyncio.run(run())
