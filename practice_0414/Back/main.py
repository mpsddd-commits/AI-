import json
import re
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from db import init_db, get_db, Post
from agents.graph import app_graph
from settings import settings

# 1. CORS 설정
origins = [
    settings.react_url,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


# 2. 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [시작 시]
    # 동기 함수인 init_db를 별도 스레드에서 실행하여 루프 차단 방지
    # 혹은 그냥 여기서 실행해도 되지만, 문제가 생기면 밖으로 빼는 게 맞습니다.
    logger.info("🚀 서버를 시작합니다...")
    
    yield  # <-- 서버가 작동하는 지점
    
    # [종료 시]
    logger.info("🛑 서버를 종료합니다. 자원을 정리합니다...")
    # 예: await 세션_풀.close() 

app = FastAPI(lifespan=lifespan)

# 만약 lifespan 안에서 init_db가 자꾸 문제를 일으킨다면 
# 그냥 여기서 실행하는 것이 정신 건강에 가장 좋습니다.
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 데이터 모델
class PromptRequest(BaseModel):
    prompt: str

# 5. 핵심 API 엔드포인트
@app.post("/chat")
async def chat(request: PromptRequest):
    logger.info(f"📩 요청 수신: {request.prompt}")
    try:
        initial_state = {"messages": [("user", request.prompt)]}
        final_state = app_graph.invoke(initial_state, config={"recursion_limit": 10})

        # 1. 도구 실행 메시지(Tool Message)를 우선적으로 찾습니다.
        # 최신 메시지부터 역순으로 탐색
        for m in reversed(final_state["messages"]):
            if m.type == 'tool':
                try:
                    # m.content가 문자열 형태의 JSON이므로 파싱
                    tool_result = json.loads(m.content)
                    
                    # 도구 리턴값이 우리가 원하는 게시글 형식을 포함하고 있다면 즉시 리턴
                    if isinstance(tool_result, dict) and "id" in tool_result:
                        return {
                            "response": {
                                "id": tool_result["id"],
                                "name": tool_result["name"],
                                "title": tool_result["title"],
                                "content": tool_result["content"]
                            }
                        }
                except:
                    # 파싱 실패 시 일반 텍스트로 처리하기 위해 패스
                    continue

        # 2. 도구 결과가 없거나 파싱 실패 시 기존 AI 답변 추출 로직 실행
        last_message = final_state["messages"][-1]
        raw_response = last_message.content
        
        # (기존 정규식 JSON 추출 로직...)
        json_match = re.search(r'(\{.*\}|\[.*\])', raw_response, re.DOTALL)
        if json_match:
            # ... 기존 코드 ...
            return {"response": json.loads(json_match.group())}

        return {"response": raw_response}

    except Exception as e:
        logger.error(f"🚨 시스템 오류 발생: {str(e)}")
        return {"response": [], "error": str(e)}
    
@app.get("/posts")
def read_posts(db: Session = Depends(get_db)):
    """
    AI를 거치지 않고 DB에서 직접 게시글 목록을 가져오는 엔드포인트
    """
    try:
        # DB에서 모든 게시글 조회 (최신순으로 정렬하고 싶다면 .order_by(Post.id.desc()) 추가)
        posts = db.query(Post).filter(Post.del_yn == True).order_by(Post.id.desc()).all()
        
        # 클라이언트가 사용하기 편하게 리스트 형태로 변환
        return [
            {
                "id": p.id, 
                "name": p.name, 
                "title": p.title, 
                "content": p.content,
                "created_at": p.created_at # 이 필드를 추가해서 프론트로 전달
            } for p in posts
        ]
    except Exception as e:
        logger.error(f"DB 조회 중 오류 발생: {e}")
        return []