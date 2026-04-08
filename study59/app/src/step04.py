from settings import settings
from src.save_image import save_graph_image
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import re

# 1. 모델 설정
model_name: str = "gemma4:e4b"
llm = ChatOllama(
  model=model_name, 
  base_url=settings.ollama_base_url,
  format="json",  # 구조화된 데이터 추출을 위해 JSON 형식 강제
  temperature=0
)

# 2. 프롬프트 템플릿 정의
# 번역 노드용 템플릿
TRANSLATION_SYSTEM_TEMPLATE = """
You are a helpful assistant that translates Korean conversation messages into English. 
Whenever a Korean message is given, provide its accurate English translation.
Do not wrap the response in any backticks or anything else. Respond with a message only!
"""

TRANSLATION_USER_TEMPLATE = """
Please translate the following Korean message into English:
{messages}
Translation Result:
"""

# 문법 교정 노드용 템플릿
GRAMMAR_CORRECTION_SYSTEM_TEMPLATE = """
You are a helpful assistant that corrects grammar in English sentences written by users in casual or conversational contexts.
Only fix clear grammar mistakes that might confuse the listener or sound unnatural in everyday conversation.
Ignore minor issues like informal punctuation, casual phrasing, or relaxed capitalization.
If the sentence is already fine for casual conversation, return it unchanged and leave feedback blank.
"""

GRAMMAR_CORRECTION_USER_TEMPLATE = """
User Input: {messages}
Feedback:
"""

# 최종 응답 노드용 템플릿
CHAT_SYSTEM_TEMPLATE = """
You are an English assistant helping users improve their grammar and conversational skills.
If the CORRECT FLAG is True, it skips without feedback and just follow up with a natural question or comment IN ENGLISH

If the CORRECT FLAG is False, you need to provide feedback on the user's sentence.
Given the corrected sentence and the feedback explanation, first explain why the sentence was corrected in a clear and friendly way IN KOREAN. 
Then, follow up with a natural question or comment IN ENGLISH that encourages the user to continue the conversation.
"""

CHAT_USER_TEMPLATE = """
Corrected Sentence: {corrected_sentence}
Feedback Explanation: {feedback}
CORRECT FLAG: {is_correct}
Answer:
"""

# 3. 데이터 구조(Pydantic & State) 정의
class GrammarFeedback(BaseModel):
  """문법 교정 결과를 담는 구조체"""
  corrected_sentence: str = Field(description="사용자의 문장을 교정한 버전")
  feedback: str = Field(description="문법적 오류에 대한 설명과 수정 이유")
  is_correct: bool = Field(description="사용자의 문장이 문법적으로 맞았는지 여부")

class State(MessagesState):
  """그래프 내부에서 공유되는 상태 객체"""
  is_correct: bool        # 문법 정답 여부
  corrected_sentence: str # 교정된 문장
  feedback: str           # 피드백 내용

# 4. 노드 함수 정의 (실제 작업 단위)

def translation(state: State):
  """한국어 입력을 영어로 번역하는 노드"""
  translation_msgs = [
    ("system", TRANSLATION_SYSTEM_TEMPLATE),
    ("user", TRANSLATION_USER_TEMPLATE),
  ]
  translation_prompt = ChatPromptTemplate.from_messages(translation_msgs)

  # 상태의 메시지 내역을 프롬프트에 주입하여 번역 수행
  response = llm.invoke(
    translation_prompt.format_messages(messages=state["messages"])
  )
  print("\n[translation node] 결과:", response.content)
  # 번역된 내용을 메시지 리스트에 추가하여 다음 노드(correction)로 전달
  return {"messages": [AIMessage(content=response.content)]}

def correction(state: State):
  """영어 문장의 문법을 검사하고 구조화된 피드백을 생성하는 노드"""
  correction_msgs = [
    ("system", GRAMMAR_CORRECTION_SYSTEM_TEMPLATE),
    ("user", GRAMMAR_CORRECTION_USER_TEMPLATE),
  ]
  correction_prompt = ChatPromptTemplate.from_messages(correction_msgs)
  
  # Pydantic 모델을 사용하여 구조화된 출력 설정
  model_with_structured_output = llm.with_structured_output(GrammarFeedback)
  
  # 가장 최근 메시지(사용자 입력 또는 번역된 결과)를 검사
  response = model_with_structured_output.invoke(
    correction_prompt.format_messages(messages=state["messages"][-1])
  )
  
  print("\n[correction node]")
  print("피드백 내용: ", response.feedback)

  # 상태 업데이트: 메시지 내역 및 문법 정보 저장
  return {
    "messages": [AIMessage(content=response.corrected_sentence)], 
    "corrected_sentence": response.corrected_sentence,
    "feedback": response.feedback,
    "is_correct": bool(response.is_correct)
  }

def respond(state: State):
  """교정 결과와 피드백을 바탕으로 사용자에게 한국어 피드백 + 영어 질문을 던지는 노드"""
  corrected_sentence = state.get("corrected_sentence", "")
  feedback = state.get("feedback", "")
  is_correct = state.get("is_correct", False)

  print("\n[respond node]")
  chat_msgs = [
    ("system", CHAT_SYSTEM_TEMPLATE),
    ("user", CHAT_USER_TEMPLATE),
  ]
  chat_prompt = ChatPromptTemplate.from_messages(chat_msgs)

  response = llm.invoke(
    chat_prompt.format_messages(
      corrected_sentence=corrected_sentence, 
      feedback=feedback, 
      is_correct=is_correct,
    )
  )
  return {"messages": [AIMessage(content=response.content)]}

# 5. 라우팅 함수
def route_function(state: State):
  """입력이 한국어면 번역 노드로, 영어면 바로 교정 노드로 분기"""
  messages = state["messages"]
  last_message = messages[-1].content

  # 정규표현식을 사용하여 한국어 포함 여부 확인
  if bool(re.search(r"[가-힣]", last_message)):
    return "translation"
  else:
    return "correction" 

# 6. 그래프 구축
def setup_graph():
  graph_builder = StateGraph(State)

  # 노드 등록
  graph_builder.add_node("translation", translation)
  graph_builder.add_node("correction", correction)
  graph_builder.add_node("respond", respond)

  # 조건부 엣지: 시작 시 언어에 따라 분기
  graph_builder.add_conditional_edges(
    START,
    route_function,
    {
      "translation": "translation",
      "correction": "correction",
    },
  )
  
  # 일반 흐름: 번역 -> 교정 -> 응답 -> 종료
  graph_builder.add_edge("translation", "correction")
  graph_builder.add_edge("correction", "respond")
  graph_builder.add_edge("respond", END)

  # 대화 기억을 위한 체크포인터 설정
  memory = MemorySaver()
  graph = graph_builder.compile(checkpointer=memory)
  
  return graph

# 7. 실행부
def run():
  try:
    graph = setup_graph()
    config = {"configurable": {"thread_id": "1"}} # 세션 아이디 설정
    
    print("번역 프로그램 시작되었습니다. (종료: q, quit, exit)")

    while True:
      try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
          print("Goodbye!")
          break

        # 그래프 실행
        response = graph.invoke({
          "messages": [{"role": "user", "content": user_input}]
        }, config=config)
        
        # 마지막 respond 노드의 결과 출력
        print("Assistant:", response["messages"][-1].content)
      except Exception as e:
          print(f"루프 내 오류: {e}")
          break

  except Exception as e:
      print(f"초기화 오류: {e}")

if __name__ == '__main__':
  run()
