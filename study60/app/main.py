from fastapi import FastAPI
from src.step01 import router as step1_router
from src.step02 import router as step2_router
from src.step03 import router as step3_router
from src.step04 import router as step4_router
from src.core import lifespan

app = FastAPI(title="LangChain Ollama Agent API", lifespan=lifespan)

# 라우터 등록
app.include_router(step1_router)
app.include_router(step2_router)
app.include_router(step3_router)
app.include_router(step4_router)
