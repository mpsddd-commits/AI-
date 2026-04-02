import requests

url = "https://pokeapi.co/api/v2/pokemon-species?limit=10000"
data = requests.get(url).json()["results"]

def main():
    try:
        KOREAN_NODE_MAP = {}

        for pokemon in data:
            detail = requests.get(pokemon["url"]).json()
            
            en_name = detail["name"].capitalize()
            
            ko_name = None
            for name in detail["names"]:
                if name["language"]["name"] == "ko":
                    ko_name = name["name"]
                    break
            
            if ko_name:
                KOREAN_NODE_MAP[en_name] = ko_name

        # 도감번호 순 정렬
        sorted_map = dict(sorted(
            KOREAN_NODE_MAP.items(),
            key=lambda x: int(requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{x[0].lower()}").json()["id"])
        ))

        print(sorted_map)
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1
    return 0

# ------------------------------------------------------
# 프로그램 실행
# ------------------------------------------------------
if __name__ == "__main__":
  exit(main())