from fastapi import APIRouter
from src.core import Query, logger
from langchain_ollama import ChatOllama
from settings import settings

router = APIRouter(
  prefix="/step02",
  tags=["영화 검색 및 AI Agent 구현"],
)

@router.post("/chat")
async def chat(query: Query):
  try:
    llm = ChatOllama(
      model=settings.ollama_model_name, 
      base_url=settings.ollama_base_url, 
    )
    message = f"당신은 영화 정보 전문가입니다. 사용자가 영화에 대해 질문하면, OMDb API를 활용하여 정보를 제공하세요. {query.input}에 대한 정보를 알려주세요."

    response = llm.invoke(message)
    return response
  except Exception as e:
    logger.error(f"실행 중 에러: {str(e)}")
