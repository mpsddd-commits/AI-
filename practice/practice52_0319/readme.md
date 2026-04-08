전체 아키텍처 (수정 버전)
🔥 핵심 흐름
CSV 파일
   ↓
Spark (대용량 정제)
   ↓
Pandas (소량 변환 / DB 적재)
   ↓
MariaDB
🏗️ 구성 요소 역할
1️⃣ Spark (정제 담당)

👉 역할:

대용량 CSV 처리

컬럼 정리

타입 변환

필터링 (승차/하차)

시간대 데이터 정제

👉 결과:

Spark DataFrame
2️⃣ Pandas (적재 담당)

👉 역할:

Spark → Pandas 변환

DB insert / bulk insert

👉 이유:

Spark JDBC보다
→ Pandas + SQLAlchemy가 훨씬 간단하고 안정적
3️⃣ MariaDB

👉 역할:

최종 저장소

4️⃣ FastAPI

👉 역할:

ETL 트리거 API

상태 확인

📦 전체 구조 그림
[FastAPI]
    │
    │ (API 호출 /etl)
    ▼
[PySpark]  ← Docker Spark Cluster
    │
    │ 정제 (filter, cast, rename)
    ▼
[Clean DataFrame]
    │
    │ toPandas()
    ▼
[Pandas]
    │
    │ to_sql / bulk insert
    ▼
[MariaDB]
🔥 핵심 설계 포인트
✔ Spark는 “정제까지만”

👉 여기서 끝

df = spark.read.csv(...)
df = df.filter(...)
df = df.withColumn(...)
✔ DB 적재는 Pandas
pdf = df.toPandas()

pdf.to_sql(
    name="seoul_metro",
    con=engine,
    if_exists="append",
    index=False
)
🚀 왜 이 구조가 좋은가
🔥 1️⃣ 디버깅 쉬움

Spark JDBC → 에러 추적 어려움

Pandas → Python에서 바로 확인 가능

🔥 2️⃣ 개발 속도 빠름

SQLAlchemy 활용 가능

트랜잭션 관리 쉬움

🔥 3️⃣ 유연성

중간 데이터 확인 가능

일부만 적재 가능

🔥 4️⃣ 실무 패턴

👉 실제로:

대용량 → Spark
적재 → Python

많이 씀

⚠️ 단점 (중요)
❗ 메모리 제한
df.toPandas()

👉 데이터 크면 터짐 💥

✅ 해결 방법
방법 1 (추천)
for batch in df.toLocalIterator():
    pdf = pd.DataFrame([batch.asDict()])
방법 2
df.limit(100000).toPandas()
방법 3 (베스트)
df.repartition(4)

👉 분할 후 처리

🧱 최종 기술 스택
✔ Backend

FastAPI

SQLAlchemy

✔ Data Processing

PySpark

✔ DB

MariaDB

✔ Infra

Docker Compose

Spark Cluster (Master + Worker)

🔥 한 줄 핵심

👉 Spark는 계산용 / Pandas는 DB용