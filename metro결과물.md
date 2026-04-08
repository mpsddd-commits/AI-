트러블 슈팅 리스트

요구사항 명세서에서 계획
판다스: 로컬 PC의 RAM 용량만 신경 쓰면 되죠.

spark적재시 : 제일 처음 csv파일을 스파크에 받아서 연동하려했는데 dockerfile이용해서 jdk.tar파일로 java설치 후 docker에서 spark 워커 opt경로폴더에 파일을 넣어서 작동하게 했는데 실패 . 나중에서야 이유를 알게 됨 왜냐하면 하둡이라는 메모리 기반 데이터베이스로 이용하여 csv파일을 받고 데이터 처리해야하는데 몰랏음. 이걸로 1일 소요함. Java 설치, 환경 변수 설정, 메모리 할당(driver-memory, executor-memory), 코어 수 제한 등 튜닝해야 할 설정이 산더미 ''에도 예민하게 반응 

그리고 feat01,02를 맡아 fastapi사용 후 react 붙이려함.

feat1 < 연도별 이용객 추이 분석	> - 데이터 집계자
"2008년~2021년 서울 지하철 이용객의
전체 성장 확인 및 거시적 흐름 파악"					

spark로 테이블링 제작이 가능하지만 인덱스 생성등이 되지 않아 비추천해서 포기 그래서 sqlalchemy이용해서 테이블 제작 endpoint1설정. 168개의 행 처리 (승차,하차 포함)

기존 적재된 테이블을 spark로 꺼내 정제 후 만들어진 새로운 테이블에 적재 작업 endpoint2

영역차트로 연별 인원수 합계(승,하,총), 전년대비 성장률, 승하차비율 균형도 준비 
구동 시간이 5초 내 완성 => 하나의 endpoint3으로 생성 ( spark를 이용 => 캐시를 사용하여 빠른 출력 처리)

JDBC_URL=jdbc:mariadb://${MARIADB_HOST}:${MARIADB_PORT}/${MARIADB_NAME}?sessionVariables=sql_mode='ANSI_QUOTES'을 사용해서 해결했다.

리액트 차트
백만명집계 => drill down 명으로 집계 했는데 백만명이라고 진행한 부분이 명으로 수정못함.

feat2 <역별 혼잡도 TOP N>

"인프라 투자 및 안전 인력 배치가 
우선적으로 필요한 HOT_SPOT 파악"

sqlalchemy이용해서 테이블 제작  endpoint1설정. 

기존 적재된 테이블을 spark로 꺼내 정제 후 만들어진 새로운 테이블에 적재 작업 endpoint2
처음에는 top 50을 기준으로 설정하고자 함. 테이블 작업에서 승차,하차, all 기준으로 시간대별로 top 50을 넣었더니 데이터가 기존 table보다 많은 데이터 500kb =>700kb로 늘어남 + 생각보다 0의 값이 나온것이 있엇다. 그래서 데이터 로딩 속도에 오래 걸린다는 판단이 들었다.

필터링이 년월일시간기준topn설정등 너무 많아 일단 구동 속도가 느림 + 최다이용객 보유역 + 하위역과의 격차배수를 지정하는데 욕심이 생겨서 전날 1등도 알려주고 싶어 해보고 진행하려햇더니 30초가 넘게 걸렸다. 그래서 50에서 20을 맥스로 설정 => 
endpoint를 2개로 설정해서 병렬처리를 하면 더 빨라질것 같다하여 endpoint 3(차트),4(kpi) 으로 생성.캐싱사용 + unpersist()로 제거

react에서
시간별 혼잡시간대 설정을 data는 보냈지만 사용 못함 (매우아쉽)

---------------------
spark와 pandas를 이용해서 데이터를 받아오고 정제, 테이블 적재에 대해 어떤점이 장점이고 단점인지를 인지하고 사용할 수 있다는 것을 알게 되었다.

테이블 적재시에 어떻게 테이블링을 하느냐에 따라 속도차이가 클 수 잇다는 것을 알게 되었다.

github사용과 react, backend연동에 대해 복습 및 react차트화를 알 수 있게 되었다.

전반적으로 react 당일로 급하게 진행하다보니 react 스타일 관련해서 제대로 정리 못해 다시 화면 조정작업 필요. 

feat 01 승하차비율 kpi 바꾸기
feat 02 시간대 인원 drilldown 구현


git filter-branch -f --env-filter '
OLD_NAME="gmin03133-dotcom"
NEW_NAME="mpsddd-commits"
NEW_EMAIL="mpsddd@gmail.com"

if [ "$GIT_AUTHOR_NAME" = "$OLD_NAME" ]
then
    export GIT_AUTHOR_NAME="$NEW_NAME"
    export GIT_AUTHOR_EMAIL="$NEW_EMAIL"
fi
if [ "$GIT_COMMITTER_NAME" = "$OLD_NAME" ]
then
    export GIT_COMMITTER_NAME="$NEW_NAME"
    export GIT_COMMITTER_EMAIL="$NEW_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags

git filter-branch -f --env-filter 'if [ "$GIT_AUTHOR_NAME" = "gmin03133-dotcom" ]; then export GIT_AUTHOR_NAME="mpsddd-commits"; export GIT_AUTHOR_EMAIL="mpsddd@gmail.com"; fi; if [ "$GIT_COMMITTER_NAME" = "gmin03133-dotcom" ]; then export GIT_COMMITTER_NAME="mpsddd-commits"; export GIT_COMMITTER_EMAIL="mpsddd@gmail.com"; fi' --tag-name-filter cat -- --branches --tags