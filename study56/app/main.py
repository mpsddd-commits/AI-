import sys
from settings import settings  # 로컬의 settings.py에서 설정값 로드
from github import Github      # GitHub API 조작을 위한 라이브러리
from mcp.server.fastmcp import FastMCP  # MCP 서버 구축을 위한 FastMCP 프레임워크

# 1. MCP 서버 인스턴스 초기화
# 서버 이름 "GitHub-Manager"는 클라이언트(VS Code 등)에서 식별용으로 사용됩니다.
mcp = FastMCP("GitHub-Manager")

# 2. 이슈 생성 도구 정의
@mcp.tool()
def create_issue(issue_title: str, issue_text: str) -> str:
  """
  지정된 GitHub 저장소에 새로운 이슈를 생성합니다.
  
  Args:
    issue_title (str): 생성할 이슈의 제목
    issue_text (str): 이슈의 상세 설명 내용
      
  Returns:
    str: 생성 성공 메시지 및 이슈 번호
  """
  try:
    # [중요] MCP는 stdout(표준 출력)을 통신 채널로 사용합니다.
    # 일반 print()를 쓰면 프로토콜이 깨질 수 있으므로, 로그는 stderr(표준 에러)로 보냅니다.
    print(f"Log: 이슈 생성 시도 중 - 제목: {issue_title}", file=sys.stderr)

    # GitHub 객체 생성 및 인증 (settings.py의 토큰 사용)
    g = Github(settings.github_token)
    
    # 설정된 저장소(예: "username/repo") 정보 가져오기
    repo = g.get_repo(settings.github_repo_name)
    
    # 실제 GitHub 서버에 이슈 생성 요청
    issue = repo.create_issue(title=issue_title, body=issue_text)
    
    print(f"Log: 이슈 #{issue.number} 생성 완료", file=sys.stderr)
    
    # 성공 시 클라이언트(LLM)에게 전달할 결과값 반환
    return f"성공적으로 이슈가 생성되었습니다. (이슈 번호: #{issue.number}, URL: {issue.html_url})"

  except Exception as e:
    # 에러 발생 시 상세 내용을 로그로 남기고 LLM에게 에러 메시지 전달
    error_msg = f"이슈 생성 중 오류 발생: {str(e)}"
    print(f"Error: {error_msg}", file=sys.stderr)
    return error_msg

# 3. 서버 실행부
if __name__ == "__main__":
  # transport='stdio'는 VS Code 확장 프로그램(Cline, Roo Code 등)과 
  # 표준 입출력을 통해 실시간으로 대화하도록 설정하는 핵심 옵션입니다.
  mcp.run(transport='stdio')
