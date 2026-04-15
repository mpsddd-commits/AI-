from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage
from agents.tools import create_post, update_or_delete_post
from agents.schemas import BoardListResponse
from settings import settings

tools = [create_post, update_or_delete_post]

class State(TypedDict):
    # Annotated[list, "append"]는 리턴되는 리스트를 기존 리스트에 합쳐줍니다.
    messages: Annotated[list, "append"]

# 1. LLM 설정 (JSON 모드 명시)
llm = ChatOllama(
    model=settings.ollama_model_name,
    base_url=settings.ollama_base_url,
    # format="json", # Ollama에게 JSON 출력을 강제함
    temperature=0
).bind_tools(tools)

def call_model(state: State):
    schema = BoardListResponse.model_json_schema()
    
#     system_prompt = (
#     "당신은 게시판 데이터 처리 기계입니다. 다음 규칙을 100% 준수하세요.\n"
#     "1. 오직 현재 사용자의 메시지에 명시된 정보(이름, 제목, 내용)만 사용하여 tools에 명시된 것 중 하나를 호출하세요.\n"
#     "2. 'testuser', '테스트'와 같은 예시 데이터는 절대 사용하지 마세요. 사용자의 입력에 이름이 없으면 '익명'이라고 적으세요.\n"
#     "3. 도구(tool) 호출은 사용자의 요청 한 번당 정확히 한 번만 수행합니다.\n"
#     f"4. 모든 처리가 끝나면 반드시 {schema} 형식의 JSON 데이터만 출력하고 즉시 종료하세요. 부연 설명은 금지합니다."
# )
#     system_prompt = (
#     "당신은 오직 데이터베이스 입력만을 수행하는 기계입니다.\n"
#     "준비되었다는 인사나 질문은 절대로 하지 마세요.\n\n"
#     "규칙:\n"
#     "1. 사용자 메시지에 글을 써달라는 요청이 있다면, tools에 명시된 것 중 하나를 호출하세요.\n"
#     "2. 도구 실행 전후에 어떠한 설명이나 문장도 덧붙이지 마세요.\n"
#     "3. 도구 실행이 완료되면, 오직 결과 데이터가 담긴 JSON만 출력하세요.\n"
#     f"4. 최종 응답 형식: {schema}"
# )
    system_prompt = ("""당신은 사용자의 모든 발언을 기록하고 관리하는 '데이터 로그 에이전트'입니다.

    1. 사용자가 자신의 이름, 상태, 생각, 혹은 단순히 어떤 사실을 말하면, 이를 나중에 확인할 수 있도록 반드시 'create_post' 도구를 사용하여 DB에 저장하세요.
    2. 사용자가 명시적으로 "글 써줘"라고 말하지 않아도, 의미 있는 정보라고 판단되면 도구를 호출해야 합니다.
    - 예: "나 사자야" -> 이름: 사자, 제목: 사용자 프로필 업데이트, 내용: 사용자가 자신을 사자라고 정의함.
    3. 저장할 때 제목(title)은 사용자의 말을 요약해서 당신이 멋지게 지어주세요.
    4. 삭제나 수정 요청이 들어오면 'update_or_delete_post' 도구를 사용하세요.
    """
                     )
    # state["messages"]를 직접 수정하지 않고, 호출 시에만 시스템 메시지를 합쳐서 보냅니다.
    # 이렇게 해야 그래프 루프 시 시스템 메시지가 중복 누적되지 않습니다.
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# 2. 그래프 구축 로직 (동일)
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))
workflow.set_entry_point("agent")

# def should_continue(state: State):
#     last_message = state["messages"][-1]
#     if last_message.tool_calls:
#         return "action"
#     return END

def should_continue(state: State):
    last_message = state["messages"][-1]
    
    # 객체이면서 tool_calls가 있는 경우
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"
    
    # 튜플인 경우에는 tool_calls가 있을 수 없으므로 종료
    return END


workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("action", "agent")

app_graph = workflow.compile()