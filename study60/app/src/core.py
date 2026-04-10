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
from src.db import save

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
  plot: str = Field(description="줄거리")  
  actors: str = Field(description="출연 배우")

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
            "type": m.get("Type"),
            "plot": m.get("Plot")
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

@tool
async def search_movie_details(query: str) -> str:
    """
    영화 제목(query)을 입력받아 검색된 영화의 상세 정보(줄거리, 배우 포함)를 반환합니다.
    """
    async with httpx.AsyncClient() as client:
        try:
            # 1. 목록 검색 (By Search)
            search_res = await client.get(
                settings.movie_api_url,
                params={"s": query, "apikey": settings.movie_api_key},
                timeout=10.0
            )
            search_res.raise_for_status()
            search_data = search_res.json()

            if search_data.get("Response") != "True":
                return json.dumps({"error": f"'{query}' 결과 없음"}, ensure_ascii=False)

            search_results = search_data.get("Search", [])
            detailed_results = []

            # 2. 검색된 결과 중 상위 몇 개에 대해 '상세 정보(By ID)'를 다시 요청
            for movie in search_results[:5]:
                movie_id = movie.get("imdbID")
                  
                # 상세 조회 API 호출 (plot=full 필수)
                detail_res = await client.get(
                    settings.movie_api_url,
                    params={
                        "i": movie_id,
                        "plot": "full", # 전체 줄거리를 가져오는 옵션
                        "apikey": settings.movie_api_key
                    }
                )
                d = detail_res.json()
                
                # 모델 필드명(소문자)과 API 응답 필드명(대문자) 매핑
                detailed_results.append({
                    "imdbID": d.get("imdbID"),
                    "title": d.get("Title"),
                    "poster": d.get("Poster"),
                    "year": d.get("Year"),
                    "type": d.get("Type"),
                    "plot": d.get("Plot"),   
                    "actors": d.get("Actors")
                })

            return json.dumps({"status": "success", "results": detailed_results}, ensure_ascii=False)

        except Exception as e:
            logger.error(f"도구 실행 중 오류: {str(e)}")
            return json.dumps({"error": "데이터를 가져오는 중 오류 발생"}, ensure_ascii=False)
        
@tool
async def insert_data(movie_data: dict) -> str:
    """
    영화 정보를 데이터베이스의 movies 테이블에 저장하거나 업데이트합니다.
    
    Args:
        movie_info (dict): 다음 키를 포함하는 사전 객체입니다:
            - imdbID (str): 영화 고유 ID
            - title (str): 영화 제목
            - poster (str): 포스터 URL
            - year (str): 개봉 연도
            - type (str): 콘텐츠 유형
            - plot (str): 전체 줄거리
            - actors (str): 출연 배우 목록
            
    Returns:
        str: 저장 성공 여부를 알리는 메시지
    """
    try:
        sql_create = f"""
              create table if not exists omdb.movies (
              imdbID VARCHAR(50) PRIMARY KEY,
              title VARCHAR(255),
              poster VARCHAR(255),
              year VARCHAR(4),
              type VARCHAR(20),
              plot VARCHAR(5000),
              actors VARCHAR(1000)
              )
              """
        save(sql_create)
        sql_insert = f"""
              INSERT INTO movies (imdbID, title, poster, year, type, plot, actors)
              VALUES ('{movie_data['imdbID']}', '{movie_data['title']}', '{movie_data['poster']}', '{movie_data['year']}', '{movie_data['type']}', '{movie_data['plot']}', '{movie_data['actors']}')
              """
        save(sql_insert)
        logger.info(f"데이터베이스에 저장된 영화 정보: {movie_data}")
        return json.dumps({"status": "success", "message": "영화 정보가 데이터베이스에 저장되었습니다."}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"데이터 저장 중 오류: {str(e)}")
        return json.dumps({"error": "데이터 저장 중 오류 발생"}, ensure_ascii=False)
   
tools = [search_movie_info, search_movie_details, insert_data]

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
    f"당신은 영화 정보 전문가입니다. 다음 절차를 엄격히 준수하세요:\n"
    f"1. search_movie_details를 사용하여 상세 정보를 가져온다.\n"
    f"2. 가져온 정보를 insert_data를 사용하여 데이터베이스에 저장한다. (반드시 호출할 것!)\n"
    f"3. 마지막으로 사용자에게 {schema} 형식에 맞춰 JSON을 출력한다.\n"
    f"설명이나 인사말 없이 오직 JSON만 출력하세요."
  )
    app.state.agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    logger.info("Agent Session Created Successfully!")
    yield
  except Exception as e:
    logger.error(f"초기화 중 오류 발생: {e}")
  finally:
    logger.info("Finalizing shutdown...")