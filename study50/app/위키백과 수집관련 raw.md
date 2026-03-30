import ollama
import json
from requests import get
from bs4 import BeautifulSoup as bs

# ---------------------------
# Ollama LLM 호출 함수
# ---------------------------
def llm_call_structured(prompt: str, model: str = "gemma3:4b"):

  final_prompt = prompt + """
  Return ONLY valid SJON. Do NOT include explanations or commentary.
  """


  # Ollama에 LLM 요청
  response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": final_prompt}],
    format={
      "type": "object",
      "properties": {
        "message" : {"type": "string"}
      },
      "required": ["message"]
    }
  )

  # 모델 응답 텍스트 추출
  text = response["message"]["content"]

  # JSON 파싱 시도
  try:
    parsed = json.loads(text)
    print(type(parsed), parsed)
  except json.JSONDecodeError:
    print("응답 오류")

# ------------------------------------------------------
# 메인 실행 함수
# ------------------------------------------------------
def main():
  try:
    episode_links = [
      "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
    ]
    options = ["Season1","Season2","Season3","Season4"]
    head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    res = get(episode_links[0], headers=head)
    if res.status_code == 200:
      episodes = []
      soup = bs(res.text)
      table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")
      rows = table.select("tr.vevent.module-episode-list-row")
      for i, row in enumerate(rows, start=1):
        synopsis = None
        synopsis_row = row.find_next_sibling("tr", class_="expand-child")
        if synopsis_row:
          synopsis_cell = synopsis_row.select_one("td.description div.shortSummaryText")
          synopsis = synopsis_cell.get_text(strip=True) if synopsis_cell else None
        episodes.append({
          "season": options[0],
          "episode_in_season": i,
          "synopsis": synopsis
        })  
    print(episodes) 

    # 프롬프트 설정
    # prompt = """
    #   안뇽하세요
    # """

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