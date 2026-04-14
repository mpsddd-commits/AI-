from pydantic import BaseModel, Field
from typing import List
# 게시판 상세 구조
class BoardDetail(BaseModel):
    name: str = Field(..., description="작성자 이름")
    title: str = Field(..., description="제목")
    content: str = Field(..., description="내용")

# 에이전트가 최종적으로 반환할 응답 구조
class BoardListResponse(BaseModel):
    success: bool = Field(..., description="작업 성공 여부")
    data: List[BoardDetail] = Field(default_factory=list, description="저장된 게시판 상세 정보 리스트")