import ollama
import json
import re
from typing import List, Optional, Dict, Union
from requests import get
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel

# 속성 타입 정의 (문자열, 숫자, bool, None)
PropertyValue = Union[str, int, float, bool, None]

# ---------------------------
# 지식 그래프 기본 모델 정의
# ---------------------------
class Node(BaseModel):
    id: str # 노드 ID (예: N0)
    label: str # 노드 타입 (예: "인간")
    properties: Optional [Dict[str, PropertyValue]] = None # 속성 딕셔너리

class Relationship(BaseModel):
    type: str # 관계 유형
    start_node_id : str # 시작 노드 ID
    end_node_id : str # 끝 노드 ID
    properties:  Optional [Dict[str, PropertyValue]] = None # 관계 속성

class KnowledgeGraph(BaseModel):
    nodes: List[Node] # 노드 리스트
    relationships: List[Relationship] # 관계 리스트


# ---------------------------
# Ollama LLM 호출 함수
# ---------------------------
def llm_call_structured(prompt: str, model: str = "gemma3:4b"):

  final_prompt = prompt + """
  Return ONLY valid JSON. Do NOT include explanations or commentary.
  """

  # Ollama에 LLM 요청
  response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": final_prompt}],
    format={
      "type": "object",
      "properties": {
        "message": {"type": "string"}
      },
      "required": ["message"]
    }
  )

  # 모델 응답 텍스트 추출
  text = response["message"]["content"]

  # JSON 파싱 시도
  try:
    parsed = json.loads(text)
    print(parsed)
  except json.JSONDecodeError:
    print("응답 오류")
    print(text)


# ------------------------------------------------------
# 데이터 처리 함수
# ------------------------------------------------------
def process_data(episodes: List[dict]):
  print("===데이터 처리 시작===")

# ------------------------------------------------------
# 위키피디아 에피소드 데이터 수집
# ------------------------------------------------------

def fetch_episode(link: str) -> List[dict]:
  season = int(re.search(r"season_(\d+)", link).group(1))  # 시즌 번호 추출
  print(f"Fetching Season {season} from: {link}")
  headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}  # 요청 헤더
  response = get(link, headers=headers)  # GET 요청
  
  soup = bs(response.text, "html.parser")  # HTML 파싱
  table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")  # 에피소드 테이블 찾기

  episodes = []
  rows = table.select("tr.vevent.module-episode-list-row")  # 각 에피소드 row

  for i, row in enumerate(rows, start=1):  # 에피소드 번호 생성
    synopsis = None
    synopsis_row = row.find_next_sibling("tr", class_="expand-child")  # 시놉시스 row 찾기
    if synopsis_row:
      synopsis_cell = synopsis_row.select_one("td.description div.shortSummaryText")
      synopsis = synopsis_cell.get_text(strip=True) if synopsis_cell else None

    episodes.append({
      "season": season,
      "episode_in_season": i,
      "synopsis": synopsis,
    })
  
  return episodes

# ------------------------------------------------------
# 메인 실행 함수
# ------------------------------------------------------
def main():
  try:
    episode_links = [
      "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
    ]
    all_episodes = []
    for link in episode_links:
      try:
        episodes = fetch_episode(link)
        all_episodes.extend(episodes)
      except Exception as e:
        print(f"Error fetching data from {link}: {e}")
        continue    
    print(all_episodes)
    print(f"총 {len(all_episodes)}개 에피소드 수집 완료")
    
    # 데이터 처리
    process_data(all_episodes)


    # 프롬프트 설정
    prompt = """
      안녕하세요
    """

    # LLM 호출
    # llm_call_structured(prompt)
  except Exception as e:
    print(f"오류 발생: {e}")
    return 1
  return 0

# ------------------------------------------------------
# 프로그램 실행
# ------------------------------------------------------
if __name__ == "__main__":
  exit(main())