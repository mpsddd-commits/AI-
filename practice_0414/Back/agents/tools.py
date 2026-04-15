from langchain_core.tools import tool
from db import SessionLocal, Post
from typing import Optional

@tool
def create_post(name: str, title: str, content: str):
    """새로운 게시글을 생성합니다. 이름, 제목, 내용은 입력받은 그대로 저장하며, AI가 임의로 수정하지 않습니다."""
    with SessionLocal() as db:
        # 1. 게시글 객체 생성 (제목 수정 없이 입력받은 그대로 저장)
        new_post = Post(name=name, title=title, content=content)
        db.add(new_post)
        
        # 2. DB에 저장
        db.commit() 
        
        # 3. 생성된 ID 등 최신 정보를 객체에 반영
        db.refresh(new_post) 

        # 4. 리액트가 바로 쓸 수 있는 딕셔너리 형태로 반환
        return {
            "id": new_post.id,
            "name": new_post.name,
            "title": new_post.title,
            "content": new_post.content,
            "message": "성공적으로 생성되었습니다."
        }

# @tool
# def list_all_posts():
#     """
#     데이터베이스(DB)에 저장된 모든 게시글 레코드를 불러오는 핵심 도구입니다.
#     사용자가 다음과 같은 요청을 할 때 반드시 이 도구를 호출해야 합니다:
#     - '게시글 목록 보여줘', '게시글 전체 목록 보여줘', '전체 글 보기'
#     - '저장된 데이터 조회', '현재 어떤 글이 있어?', '데이터 로그 출력'
#     - 아무런 정보 없이 '목록'이라고만 말할 때
#     이 함수는 리스트 형태의 데이터를 반환하며, AI는 이 데이터를 바탕으로 최종 JSON 응답을 구성해야 합니다.
#     """
#     with SessionLocal() as db:
#         try:
#             posts = db.query(Post).all()
#             if not posts: 
#                 return "현재 데이터베이스에 저장된 게시글이 하나도 없습니다. 비어있는 상태입니다."
            
#             # 리스트 객체를 그대로 리턴 (JSON 직렬화는 모델/그래프가 처리)
#             return [{"id": p.id, "title": p.title, "name": p.name} for p in posts]
#         except Exception as e:
#             return f"데이터 조회 중 에러 발생: {str(e)}"
# 이것은 해본 결과 너무 오래 걸리고 AI가 잘 받아들이지 못해서 포기

@tool
def update_or_delete_post(action: str, post_id: int, title: Optional[str] = None, content: Optional[str] = None):
    """
    게시글을 수정하거나 삭제하는 도구입니다.
    - action: 'update' (수정) 또는 'delete' (삭제) 중 하나를 입력해야 합니다.
    - post_id: 대상이 되는 게시글의 고유 ID(숫자)입니다.
    - 수정 시에는 수정할 title이나 content를 함께 제공하세요.
    사용자가 '3번 글 지워줘'라고 하면 action='delete', post_id=3으로 호출합니다.
    """
    with SessionLocal() as db:
        post = db.query(Post).filter(Post.id == post_id).first()
        if not post: return "해당 ID의 글을 찾을 수 없습니다."
        
        if action == "delete":
            post.del_yn = False
            db.commit()
            return f"ID {post_id} 삭제 완료."
        elif action == "update":
            if title: post.title = title
            if content: post.content = content
            db.commit()
            return f"ID {post_id} 수정 완료."