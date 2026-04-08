from src import step01, step02, step03, step04, step05, step06, step07, step08, step09, step10

models = [
  "llama3.2:3b", "gemma3:4b", "gemma4:e4b", "gpt-oss:20b", "qwen3:8b"
]
embed_models = [
  "mxbai-embed-large:latest", "nomic-embed-text:v1.5", "nomic-embed-text-v2-moe:latest", "qwen3-embedding:8b"
]

def main():
  print("Hello from app!")
  # print("Step 1 실행: Ollama 모델과의 통신을 위한 간단한 함수")
  # step01.run()
  # print("Step 2 실행: Ollama 모델과의 스트리밍(Streaming) 통신을 위한 간단한 함수")
  # step02.run(models[2])
  # print("Step 3 실행: Ollama 모델과의 통신을 위한 체인(Chain) 구성")
  # step03.run(models[2])
  # print("Step 4 실행: Ollama Embedding(임베딩) 모델과 RAG(Retrieval-Augmented Generation, 검색 증강 생성) => rag 사용 이유는 파인튜닝하기에는 소모값이 너무 크므로 웹서칭 등을 통해 rag 사용 . 단점으로는 속도가 느리다. 그래서 속도를 빠르게 하기 위해 파인튜닝 사용 모델은 있고 내용만 조금 추가하는 것 ")
  # step04.run(models[2])
  # print("Step 5 실행: Ollama Embedding(임베딩) 모델과 RAG(Retrieval-Augmented Generation, 검색 증강 생성) 기술에 스트리밍(Streaming)")
  # step05.run()
  # print("Step 6 실행: Ollama 채팅 모델 만들기")
  # step06.run(models[2])
  # print("Step 7 실행: Ollama LLM(모델)이 tool(도구)를 호출하도록 만들기") => tool == skill 오픈클로에서 말하는 + 랭체인 완료
  # step07.run(models[2])
  # print("Step 8 실행: 그래프 상태값을 업데이트하는 노드 만들기")
  # step08.run()
  # print("Step 9 실행: 그래프 상태값을 업데이트하기")
  # step09.run()
  # print("Step 10 실행: 그래프 상태값을 Annotated와 operator.add를 이용한 상태 업데이트(Reducer)")
  # step10.run()

if __name__ == "__main__":
  main()
