from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean 
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func # func 추가
from settings import settings

# 1. DB 연결 설정
MARIADB_URL = settings.mariadb_url
engine = create_engine(MARIADB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. 테이블 모델 정의
class Post(Base):
    __tablename__ = "hyun_post"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    del_yn = Column(Boolean, default=True)

# 3. 테이블 생성 함수
def init_db():
    Base.metadata.create_all(bind=engine)

# 4. FastAPI용 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()