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
    getattr(settings, "react_url", "http://localhost:5173"),
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
        # [중요] recursion_limit을 설정하여 무한 루프(무한 대기)를 방지합니다.
        # 도구 실행(1회) -> 결과 보고(1회) 정도면 충분하므로 10으로 제한합니다.
        initial_state = {"messages": [("user", request.prompt)]}
        final_state = app_graph.invoke(
            initial_state, 
            config={"recursion_limit": 10} 
        )
        for m in final_state["messages"]:
            if hasattr(m, 'tool_calls'):
                print(f"🛠️ 도구 호출 시도: {m.tool_calls}")
            if m.type == 'tool':
                print(f"✅ 도구 실행 결과: {m.content}")
                
        # 마지막 AI 메시지 추출
        last_message = final_state["messages"][-1]
        raw_response = last_message.content
        
        print(f"🔍 AI 원본 응답 로그:\n{raw_response}\n{'-'*30}")

        # 1. JSON 패턴 추출 (정규식)
        # AI가 앞뒤에 설명을 붙여도 JSON만 골라냅니다.
        json_match = re.search(r'(\{.*\}|\[.*\])', raw_response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            try:
                structured_data = json.loads(json_str)
                # AI가 'data' 키를 썼는지, 아니면 리스트 자체인지 확인
                if isinstance(structured_data, dict):
                    content = structured_data.get("data", structured_data.get("response", []))
                    # 만약 dict인데 위 키들이 없다면 dict 자체를 리스트에 담거나 조사 필요
                    return {"response": content}
                return {"response": structured_data} # 리스트인 경우
            except json.JSONDecodeError:
                pass

            # 만약 JSON 파싱에 실패했지만, AI가 도구를 실행했다면 
            # final_state["messages"]를 뒤져서 도구 결과를 강제로 추출할 수도 있습니다.
            for m in reversed(final_state["messages"]):
                if m.type == 'tool':
                    try:
                        # 도구 실행 결과가 리스트 형태의 문자열이라면 파싱
                        return {"response": json.loads(m.content)}
                    except:
                        return {"response": m.content}

            return {"response": [], "error": "목록을 형식에 맞게 가져오지 못했습니다."}

    except Exception as e:
        logger.error(f"🚨 시스템 오류 발생: {str(e)}")
        # 무한 루프나 타임아웃 발생 시 프론트엔드에 에러 반환
        return {
            "response": [],
            "error": "AI가 응답을 생성하는 중에 시간이 초과되었거나 오류가 발생했습니다.",
            "details": str(e)
        }
    
@app.get("/posts")
def read_posts(db: Session = Depends(get_db)):
    """
    AI를 거치지 않고 DB에서 직접 게시글 목록을 가져오는 엔드포인트
    """
    try:
        # DB에서 모든 게시글 조회 (최신순으로 정렬하고 싶다면 .order_by(Post.id.desc()) 추가)
        posts = db.query(Post).order_by(Post.id.desc()).all()
        
        # 클라이언트가 사용하기 편하게 리스트 형태로 변환
        return [
            {
                "id": p.id, 
                "name": p.name, 
                "title": p.title, 
                "content": p.content
            } for p in posts
        ]
    except Exception as e:
        logger.error(f"DB 조회 중 오류 발생: {e}")
        return []