import ollama
import json
import re
import os
from typing import List, Optional, Dict, Union
from requests import get
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel
from node import UPDATED_TEMPLATE, KOREAN_NODE_MAP

# 속성 타입 정의 (문자열, 숫자, bool, None)
PropertyValue = Union[str, int, float, bool, None]

# ---------------------------
# 지식 그래프 기본 모델 정의
# ---------------------------
class Node(BaseModel):
  id: str  # 노드 ID (예: N0)
  label: str  # 노드 타입 (예: "인간")
  properties: Optional[Dict[str, PropertyValue]] = None  # 속성 딕셔너리

class Relationship(BaseModel):
  type: str  # 관계 유형
  start_node_id: str  # 시작 노드 ID
  end_node_id: str  # 끝 노드 ID
  properties: Optional[Dict[str, PropertyValue]] = None  # 관계 속성

class GraphResponse(BaseModel):
  nodes: List[Node]  # 노드 리스트
  relationships: List[Relationship]  # 관계 리스트

# ---------------------------
# Ollama LLM 호출 함수
# ---------------------------
def llm_call_structured(prompt: str, model: str = "llama3.1:latest") -> GraphResponse:

  final_prompt = prompt + """
  Return ONLY valid JSON. Do NOT include explanations or commentary.
  """

  # Ollama에 LLM 요청
  response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": final_prompt}]
  )

  # 모델 응답 텍스트 추출
  text = response["message"]["content"]

  # JSON 파싱 시도
  try:
    parsed = json.loads(text)
    print(parsed)
  except json.JSONDecodeError:
    # 전체 텍스트에서 JSON 블록만 추출
    json_text = re.search(r"\{.*\}", text, re.S)
    if not json_text:
      raise Exception("모델 응답에서 JSON을 찾지 못했습니다.")
    parsed = json.loads(json_text.group(0))
  
  return GraphResponse(**parsed)  # pydantic 모델로 변환 후 반환

# ------------------------------------------------------
# 여러 에피소드 그래프를 통합하기 위한 함수
# ------------------------------------------------------
def combine_chunk_graphs(chunk_graphs: list) -> GraphResponse:
  all_nodes = []  # 모든 노드를 담을 리스트
  for chunk_graph in chunk_graphs:
    for node in chunk_graph.nodes:
      all_nodes.append(node)
  
  all_relationships = []  # 모든 관계를 담을 리스트
  for chunk_graph in chunk_graphs:
    for relationship in chunk_graph.relationships:
      all_relationships.append(relationship)
  
  unique_nodes = []  # 중복 제거된 최종 노드 리스트
  seen = set()  # 노드 중복 체크용

  for node in all_nodes:
    node_key = (node.id, node.label, str(node.properties))  # 노드 고유값 생성
    if node_key not in seen:
      unique_nodes.append(node)
      seen.add(node_key)

  return GraphResponse(nodes=unique_nodes, relationships=all_relationships)

# ------------------------------------------------------
# 수집된 데이터를 LLM으로 처리하여 그래프 생성
# ------------------------------------------------------
def process_data(episodes: List[dict]) -> GraphResponse:
    print("\n" + "="*30)
    print("      지식 추출 및 변환 시작      ")
    print("="*30)

    chunk_graphs: List[GraphResponse] = []
    
    for episode in episodes:
        if not episode.get("synopsis"):
            continue
            
        ep_id = f"S{episode['season']}E{episode['episode_in_season']:02d}"
        print(f"\n▶ [{ep_id}] 처리 중...")
        
        try:
            prompt = UPDATED_TEMPLATE + f"\n 입력값\n {episode['synopsis']}"
            graph_response = llm_call_structured(prompt)

            # 노드 이름 변환 및 로그 출력
            converted_nodes = 0
            for node in graph_response.nodes:
                english_name = node.properties.get("name", "")
                if english_name in KOREAN_NODE_MAP:
                    old_name = english_name
                    new_name = KOREAN_NODE_MAP[english_name]
                    node.properties["name"] = new_name
                    converted_nodes += 1
            
            # 관계 속성 추가 및 요약 출력
            for relationship in graph_response.relationships:
                if relationship.properties is None:
                    relationship.properties = {}
                relationship.properties["episode_number"] = ep_id
            
            # [추가] 처리 결과 요약 프린트
            print(f"   └ ✨ 추출 성공: 노드 {len(graph_response.nodes)}개, 관계 {len(graph_response.relationships)}개")
            if converted_nodes > 0:
                print(f"   └ 🇰🇷 한글 변환: {converted_nodes}개의 노드 이름이 번역되었습니다.")
            
            # 관계 예시 출력 (첫 번째 관계만)
            if graph_response.relationships:
                rel = graph_response.relationships[0]
                print(f"   └ 🔗 관계 예시: {rel.start_node_id} --({rel.type})--> {rel.end_node_id}")

            chunk_graphs.append(graph_response)
            
        except Exception as e:
            print(f"   └ ❌ 오류 발생: {e}")
            continue
    
    if not chunk_graphs:
        raise Exception("그래프를 성공적으로 추출하지 못했습니다.")
    
    print("\n" + "="*30)
    print(f"✅ 총 {len(chunk_graphs)}개 에피소드 분석 완료")
    print("="*30 + "\n")
    
    return combine_chunk_graphs(chunk_graphs)

# ------------------------------------------------------
# 위키피디아 에피소드 데이터 수집
# ------------------------------------------------------
def fetch_episode(link: str, season_idx: int) -> List[dict]:
  season = season_idx
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
# 출력 파일 저장
# ------------------------------------------------------

def save_output(episodes: List[dict], final_graph: GraphResponse):
  print("=== 결과 저장 ===")
  
  os.makedirs("output", exist_ok=True)  # output 폴더 생성
  
  with open("output/1_원본데이터.json", "w", encoding="utf-8") as f:
      json.dump(episodes, f, indent=2, ensure_ascii=False)
  print("원본 데이터 저장: output/1_원본데이터.json")
  
  with open("output/지식그래프_최종.json", "w", encoding="utf-8") as f:
      json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)
  print("최종 지식그래프 저장: output/지식그래프_최종.json")

# ------------------------------------------------------
# 메인 실행 함수
# ------------------------------------------------------
def main():
  try:
    episode_links = [
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Indigo_League",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Adventures_in_the_Orange_Islands",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_The_Johto_Journeys",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Johto_League_Champions",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Master_Quest",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Advanced",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Advanced_Challenge",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Advanced_Battle",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Battle_Frontier",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_Diamond_and_Pearl",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Diamond_and_Pearl:_Battle_Dimension",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Diamond_and_Pearl:_Galactic_Battles",
      "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Diamond_and_Pearl:_Sinnoh_League_Victors",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_Black_%26_White",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Black_%26_White:_Rival_Destinies",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Black_%26_White:_Adventures_in_Unova_and_Beyond",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_XY",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_XY_Kalos_Quest",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_XYZ",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_Sun_%26_Moon",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_Sun_%26_Moon_%E2%80%93_Ultra_Adventures",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_the_Series:_Sun_%26_Moon_%E2%80%93_Ultra_Legends",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_Journeys:_The_Series",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_Master_Journeys:_The_Series",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_Ultimate_Journeys:_The_Series",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_Horizons:_The_Series",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_Horizons_%E2%80%93_The_Search_for_Laqua",
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon_Horizons_%E2%80%93_Rising_Hope",
    ]
    all_episodes = []
    for i, link in enumerate(episode_links, start=1):
          try:
            episodes = fetch_episode(link, i)
            all_episodes.extend(episodes)
          except Exception as e:
            print(f"Error fetching data from {link}: {e}")
            continue
    print(f"총 {len(all_episodes)}개 에피소드 수집 완료")

    final_graph = process_data(all_episodes)

    save_output(episodes, final_graph)  # 결과 저장
        
    print("=" * 50)
    print("✅ 지식그래프 생성 완료!")
    print(f"📊 총 노드 수: {len(final_graph.nodes)}")
    print(f"🔗 총 관계 수: {len(final_graph.relationships)}")    
  except Exception as e:
    print(f"오류 발생: {e}")
    return 1
  return 0

# ------------------------------------------------------
# 프로그램 실행
# ------------------------------------------------------
if __name__ == "__main__":
  exit(main())