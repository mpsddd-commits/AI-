import logging
import json
import re
import httpx
from settings import settings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Query(BaseModel):
  input: str

class MovieItem(BaseModel):
  imdbID: str = Field(description="영화 고유 ID") 
  title: str = Field(description="제목")
  poster: str = Field(description="포스터 이미지 URL")
  year: str = Field(description="개봉 년도")
  type: str = Field(description="유형 (movie, series 등)")

class MovieListResponse(BaseModel):
  movies: list[MovieItem] = Field(description="검색된 영화 리스트")
  count: int = Field(description="검색된 총 영화 개수")

def get_app_state(request: Request):
  return request.app.state

def extract_json(text: str) -> dict:
  match = re.search(r"(\{.*\})", text, re.DOTALL)
  if match:
      return json.loads(match.group(1))
  return json.loads(text)

@tool
async def search_movie_info(query: str) -> str:
  """
  영화 제목(query)을 입력받아 검색된 영화들의 리스트를 JSON 형식의 문자열로 반환합니다.
  반환 구조: [{'imdbID': ..., 'title': ..., 'poster': ..., 'year': ..., 'type': ...}, ...]
  """
  async with httpx.AsyncClient() as client:
    try:
      response = await client.get(
        settings.movie_api_url,
        params={"s": query, "apikey": settings.movie_api_key},
        timeout=10.0
      )
      response.raise_for_status()
      data = response.json()

      if data.get("Response") == "True":
        search_results = data.get("Search", [])
        formatted_data = [
          {
            "imdbID": m.get("imdbID"),
            "title": m.get("Title"),
            "poster": m.get("Poster"),
            "year": m.get("Year"),
            "type": m.get("Type")
          } for m in search_results
        ]
        return json.dumps(formatted_data, ensure_ascii=False)
      else:
        return json.dumps({"error": f"'{query}'에 대한 검색 결과가 없습니다."}, ensure_ascii=False)

    except httpx.HTTPStatusError as e:
      logger.error(f"API 요청 오류: {e.response.status_code}")
      return json.dumps({"error": "영화 서버 응답 오류가 발생했습니다."}, ensure_ascii=False)
    except Exception as e:
      logger.error(f"예상치 못한 오류: {str(e)}")
      return json.dumps({"error": "네트워크 연결이 원활하지 않습니다."}, ensure_ascii=False)

tools = [search_movie_info]

@asynccontextmanager
async def lifespan(app: FastAPI):
  try:
    llm = ChatOllama(
      model=settings.ollama_model_name, 
      base_url=settings.ollama_base_url, 
      format="json",
      temperature=0
    )
    schema = MovieListResponse.model_json_schema()
    system_message = (
      f"당신은 영화 정보 전문가입니다. 반드시 search_movie_info 도구를 사용해 정보를 찾으세요. "
      f"응답은 반드시 다음 JSON 스키마를 따르는 순수한 JSON 객체여야 합니다: {schema}. "
      f"설명이나 인사말 없이 JSON만 출력하세요."
    )
    app.state.agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    logger.info("Agent Session Created Successfully!")
    yield
  except Exception as e:
    logger.error(f"초기화 중 오류 발생: {e}")
  finally:
    logger.info("Finalizing shutdown...")