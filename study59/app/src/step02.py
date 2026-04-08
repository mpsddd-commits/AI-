from typing import Annotated
from operator import add
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from src.save_image import save_graph_image

# 1. 상태(State) 정의: 그래프 전체에서 유지되고 전달될 데이터 구조
class State(TypedDict):
  user_input: str  # 사용자가 입력한 할 일 내용
  flag: str        # 수행할 동작 (add: 추가, complete: 완료)
  
  # 일반적인 리스트: 노드에서 반환하는 값으로 전체가 '교체(Overwrite)'됨
  todo_list: list[str] 
  
  # Annotated와 add를 사용한 리스트: 노드에서 반환하는 값이 기존 리스트에 '추가(Append)'됨
  # 즉, 덮어쓰지 않고 기록을 누적할 때 사용 (Reducer 기능)
  completed_list: Annotated[list[str], add]

# 2. 노드(Node) 함수: 상태를 받아 로직을 수행하고 업데이트된 상태를 반환
def update_todo_list(state: State) -> State:
  """
  현재 상태의 할 일 목록을 수정하고 완료된 항목을 기록합니다.
  """
  user_input = state["user_input"]
  flag = state["flag"]
  todo_list = state["todo_list"]
  
  # 'add' 플래그일 경우: 슬래시(/)로 구분된 입력값을 리스트에 확장 추가
  if flag == "add":
    todo_list.extend(user_input.split("/"))
      
  # 'complete' 플래그일 경우: 리스트에 해당 항목이 있으면 제거
  elif flag == "complete": 
    if user_input in todo_list:
      todo_list.remove(user_input)
  
  # 업데이트된 값들을 반환
  # completed_list의 경우 'add' 리듀서에 의해 기존 데이터 뒤에 붙게 됨
  return {
    "user_input": user_input, 
    "flag": flag, 
    "todo_list": todo_list, 
    "completed_list": [user_input] if flag == "complete" else []
  }

# 3. 그래프 설정 함수
def setup_graph():
  # State 구조를 사용하는 그래프 빌더 초기화
  graph_builder = StateGraph(State)
  
  # "update_todo"라는 이름으로 노드 추가
  graph_builder.add_node("update_todo", update_todo_list)
  
  # 시작 지점을 "update_todo" 노드로 설정
  graph_builder.set_entry_point("update_todo")

  # 그래프 컴파일 (실행 가능한 객체 생성)
  graph = graph_builder.compile()
  
  # (선택 사항) 그래프 시각화 이미지 저장
  # save_graph_image(graph)
  return graph

# 4. 그래프 실행 함수: 현재 상태와 입력을 그래프에 전달
def stream_graph_updates(user_input: str, flag: str, state: State, graph: StateGraph):
  """
  사용자 입력을 바탕으로 그래프를 실행(invoke)하고 결과 상태를 반환합니다.
  """
  return graph.invoke(
    {
      "user_input": user_input, 
      "flag": flag, 
      "todo_list": state["todo_list"], 
      "completed_list": state["completed_list"]
    }
  )

# 5. 메인 실행 루프
def run():
  # 그래프 초기화
  graph = setup_graph()
  
  # 초기 빈 상태 설정
  state = State(user_input="", flag="", todo_list=[], completed_list=[])
  
  print("할일 관리가 시작되었습니다. (추가: add, 완료: complete, 종료: q, quit, exit)")
  
  while True:
    try:
      # 1. 할 일 내용 입력
      user_input = input("User (할일 내용): ")
      if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
      
      # 2. 동작 플래그 입력 (add 또는 complete)
      flag = input("Flag (add/complete): ")
      
      # 3. 그래프 실행 및 상태 업데이트
      state = stream_graph_updates(user_input, flag, state, graph)
      
      # 4. 결과 출력 (현재 할 일 목록과 누적 완료 목록 확인 가능)
      print("\n--- 현재 상태 ---")
      print(f"Todo List: {state['todo_list']}")
      print(f"Completed History: {state['completed_list']}")
      print("----------------\n")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == '__main__':
  run()
