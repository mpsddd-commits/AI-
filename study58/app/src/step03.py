from src.save_image import save_graph_image
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 1. 입력 전용 상태 정의
# 그래프가 처음 실행될 때 외부에서 받아야 하는 데이터의 형식을 정의합니다.
class InputState(TypedDict):
  question: str

# 2. 출력 전용 상태 정의
# 그래프 실행이 끝난 후 사용자에게 최종적으로 반환할 데이터의 형식을 정의합니다.
class OutputState(TypedDict):
  answer: str

# 3. 전체 관리 상태 (Overall State)
# 입력(InputState)과 출력(OutputState)을 모두 상속받아, 
# 그래프 내부 노드들이 자유롭게 읽고 쓸 수 있는 전체 데이터 바구니를 만듭니다.
class OverallState(InputState, OutputState):
  pass

# 4. 노드 함수 정의
# 입력으로 InputState(질문)를 받고, 반환값으로 answer를 포함한 딕셔너리를 내보냅니다.
def answer_node(state: InputState):
  print(f"--- 질문 접수: {state['question']} ---")
  # 내부적으로는 OverallState 구조에 맞춰 데이터를 업데이트합니다.
  return {"answer": "bye", "question": state["question"]}

def run():
  # 5. 그래프 빌더 초기화 (상태 분리 설정)
  # state_schema=OverallState: 내부에서 관리할 전체 데이터 구조
  # input=InputState: 외부에서 넣을 수 있는 데이터 제한
  # output=OutputState: 최종 결과물에서 보여줄 데이터 필드 제한
  graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)
  
  # 6. 노드 등록 및 연결
  # 함수 이름 자체가 노드의 이름이 됩니다 ("answer_node").
  graph_builder.add_node(answer_node)
  graph_builder.add_edge(START, "answer_node")
  graph_builder.add_edge("answer_node", END)
  
  # 7. 그래프 컴파일
  graph = graph_builder.compile()

  # 구조 이미지 저장 (기존 작성하신 커스텀 함수 호출)
  save_graph_image(graph)

  # 8. 그래프 실행 (Invoke)
  # InputState 형식에 맞춰 {"question": "hi"}를 넣습니다.
  # OutputState 설정 덕분에 결과값에는 'answer' 필드 위주로 반환됩니다.
  result = graph.invoke({"question": "hi"})
  print("최종 응답:", result)

if __name__ == "__main__":
  run()
