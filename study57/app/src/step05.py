from settings import settings
# 텍스트를 적절한 크기로 자르는 도구
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 데이터를 벡터 형태로 저장하고 검색하게 해주는 Vector Store (FAISS)
from langchain_community.vectorstores import FAISS
# Ollama를 통한 LLM 및 임베딩(텍스트의 수치화) 모델 연결
from langchain_ollama import OllamaLLM, OllamaEmbeddings

def run(model:str = "llama3.2:3b",  question:str = "Ollama는 무엇인가?", 
  embed_model:str = "mxbai-embed-large", text:str = "Ollama는 로컬 LLM 서버입니다"):
  
  # 1. 모델 설정
  # llm: 답변 생성을 담당하는 메인 언어 모델
  # embed: 텍스트를 벡터(수치)로 변환하여 검색 가능하게 만드는 임베딩 모델
  llm = OllamaLLM(model=model, base_url=settings.ollama_base_url)
  embed = OllamaEmbeddings(model=embed_model, base_url=settings.ollama_base_url)
  
  # 2. 텍스트 분할 (Chunking)
  # 문장이 너무 길면 검색 정확도가 떨어지므로, 200자 단위로 자릅니다.
  # chunk_overlap: 앞뒤 조각을 20자씩 겹치게 하여 문맥이 끊기는 것을 방지합니다.
  splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
  docs = splitter.split_text(text)
  
  # 3. 벡터 DB 생성 및 검색 (Indexing & Retrieval)
  # 분할된 텍스트를 임베딩하여 메모리 기반 벡터 저장소(FAISS)에 저장합니다.
  db = FAISS.from_texts(docs, embed)
  
  # 질문(question)과 의미적으로 가장 유사한 텍스트 조각을 DB에서 찾아옵니다.
  retrieved_docs = db.similarity_search(question)
  
  # 검색된 문서 조각들을 하나의 문자열(context)로 합칩니다.
  context = "\n".join([d.page_content for d in retrieved_docs])
  
  # 4. 프롬프트 엔지니어링
  # 검색된 정보(context)를 프롬프트에 포함시켜 AI에게 전달할 '참고 자료'를 만들어줍니다.
  prompt = f"""
  다음 문맥을 기반으로 질문에 답하세요.

  문맥:
  {context}

  질문:
  {question}
  """
  
  # 5. 스트리밍 답변 출력
  # 사용자에게 답변 중임을 알리고, 실시간으로 답변 조각을 받아와 출력합니다.
  print(f"답변 중 ({model}): ", end="", flush=True)
  
  # llm.stream()은 전체 답변이 완성될 때까지 기다리지 않고, 생성되는 즉시 반환합니다.
  for chunk in llm.stream(prompt):
    # end="", flush=True를 통해 한 글자씩 타이핑되는 효과를 줍니다.
    print(chunk, end="", flush=True)
  
  print() # 마지막 줄바꿈

if __name__ == "__main__":
  run()