from settings import settings
# 텍스트를 적절한 크기로 자르는 도구
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 데이터를 벡터 형태로 저장하고 검색하게 해주는 Vector Store (FAISS)
from langchain_community.vectorstores import FAISS
# Ollama를 통한 LLM 및 임베딩(텍스트의 수치화) 모델 연결
from langchain_ollama import OllamaLLM, OllamaEmbeddings

def run(model:str = "llama3.2:3b",  question:str = "Ollama는 무엇인가?", 
  embed_model:str = "nomic-embed-text-v2-moe:latest", text:str = "Ollama는 로컬 LLM 서버입니다"):
  
  # 1. 모델 초기화
  # 답변을 생성할 LLM과 텍스트를 벡터로 변환할 임베딩 모델을 설정합니다.
  llm = OllamaLLM(model=model)
  embed = OllamaEmbeddings(model=embed_model, base_url=settings.ollama_base_url)
  
  # 2. 텍스트 분할 (Text Splitting)
  # 너무 긴 텍스트는 모델이 처리하기 어렵거나 검색 효율이 떨어지므로 작게 자릅니다.
  # chunk_size: 최대 200자 단위로 자름
  # chunk_overlap: 문맥 유지를 위해 앞뒤 조각이 20자 정도 겹치게 함 ( 몇 퍼센트 기준으로 하는것이 좋다)
  splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
  docs = splitter.split_text(text)
  
  # 3. 벡터 데이터베이스 구축 (Vector Store)
  # 분할된 텍스트(docs)를 임베딩 모델(embed)을 통해 숫자로 변환하여 메모리 상의 FAISS DB에 저장합니다.
  db = FAISS.from_texts(docs, embed)
  
  # 4. 관련 문서 검색 (Retrieval)
  # 사용자의 질문과 가장 유사한 의미를 가진 텍스트 조각을 DB에서 찾아옵니다.
  retrieved_docs = db.similarity_search(question)
  
  # 5. 문맥(Context) 생성
  # 검색된 문서들의 내용을 하나로 합쳐서 모델에게 줄 참고 자료를 만듭니다.
  context = "\n".join([d.page_content for d in retrieved_docs])
  
  # 6. 프롬프트 구성 (Prompt Engineering)
  # 모델에게 "네 지식뿐만 아니라 내가 준 이 문맥을 바탕으로 답해줘"라고 지시합니다.
  prompt = f"""
  다음 문맥을 기반으로 질문에 답하세요.

  문맥:
  {context}

  질문:
  {question}
  """
  
  # 7. 답변 생성 및 출력
  # 참고 자료가 포함된 프롬프트를 LLM에 전달하여 최종 답변을 얻습니다.
  response = llm.invoke(prompt)
  print(response)

if __name__ == "__main__":
  run()
