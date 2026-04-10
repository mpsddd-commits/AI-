from fastapi import APIRouter, HTTPException, Depends
from src.core import Query, logger
from settings import settings
import httpx
import json

router = APIRouter(
  prefix="/step01",
  tags=["영화 검색"],
)

@router.post("/chat")
async def chat(query: Query):
  async with httpx.AsyncClient() as client:
    try:
      response = await client.get(
        settings.movie_api_url,
        params={"s": query.input, "apikey": settings.movie_api_key},
        timeout=10.0
      )
      response.raise_for_status()
      data = response.json()
      detailed_results = []

      if data.get("Response") == "True":
        search_results = data.get("Search", [])
        for movie in search_results[:3]:  # 너무 많으면 느려지므로 상위 3개만 예시
                    movie_id = movie.get("imdbID")
                    
                    detail_response = await client.get(
                        settings.movie_api_url,
                        params={
                            "i": movie_id, 
                            "plot": "full",  # <--- 줄거리 전체 설명을 가져오는 핵심 설정
                            "apikey": settings.movie_api_key
                        }
                    )
                    detail_data = detail_response.json()
                    detailed_results.append(detail_data)
        return {"status": "success", "results": detailed_results}
    except httpx.HTTPStatusError as e:
      logger.error(f"API 요청 오류: {e.response.status_code}")
      return json.dumps({"error": "영화 서버 응답 오류가 발생했습니다."}, ensure_ascii=False)
    except Exception as e:
      logger.error(f"예상치 못한 오류: {str(e)}")
      return json.dumps({"error": "네트워크 연결이 원활하지 않습니다."}, ensure_ascii=False)
