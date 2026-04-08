from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace, when, to_date, dayofweek
from fastapi import FastAPI
from settings import settings

app = FastAPI()


# ==============================
# Spark 시작
# ==============================
@app.on_event("startup")
def startup():
    spark = SparkSession.builder \
        .appName("SparkETL") \
        .master(settings.spark_host) \
        .config("spark.jars", settings.jar_path) \
        .config("spark.driver.host", settings.host_ip) \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.driver.port", "10000") \
        .config("spark.blockManager.port", "10001") \
        .config("spark.cores.max", "2") \
        .config("spark.sql.sources.jdbc.driver.kind", "mariadb") \
        .config("spark.sql.dialect", "mysql") \
        .getOrCreate()

    app.state.spark = spark
    print("✅ Spark 시작 완료")


@app.on_event("shutdown")
def shutdown():
    spark = getattr(app.state, "spark", None)
    if spark:
        spark.stop()


# ==============================
# Spark ETL
# ==============================
@app.post("/etl")
def run_etl():
    spark = getattr(app.state, "spark", None)

    if not spark:
        return {"status": False, "error": "Spark not initialized"}

    try:
        # 1️⃣ CSV 읽기 (Spark)
        df = spark.read \
            .option("header", True) \
            .option("encoding", "UTF-8") \
            .csv(settings.file_dir)

        # 2️⃣ 컬럼 공백 제거
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip())

        # 3️⃣ 필요없는 컬럼 제거
        drop_cols = ["연번", "호선", "합계"]
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(c)

        # 4️⃣ 구분 컬럼 정리
        if "구분.1" in df.columns:
            df = df.drop("구분")
            df = df.withColumnRenamed("구분.1", "구분")

        # 5️⃣ 날짜 처리
        if "날짜" in df.columns:
            df = df.withColumn("날짜", to_date(col("날짜"), "yyyy-MM-dd"))
            df = df.withColumn("요일", dayofweek(col("날짜")))

        # 6️⃣ 구분 필터
        df = df.withColumn("구분", trim(col("구분")))
        df = df.filter(col("구분").isin("승차", "하차"))

        # 7️⃣ 시간 컬럼 숫자 변환
        for c in df.columns:
            if "~" in c:
                df = df.withColumn(
                    c,
                    regexp_replace(col(c), ",", "").cast("int")
                )

        # 8️⃣ 필요한 컬럼만 선택
        TARGET_COLUMNS = [
            "날짜", "요일", "역번호", "역명", "구분",
            "05~06","06~07","07~08","08~09","09~10",
            "10~11","11~12","12~13","13~14","14~15",
            "15~16","16~17","17~18","18~19","19~20",
            "20~21","21~22","22~23","23~24","24~"
        ]

        for c in TARGET_COLUMNS:
            if c not in df.columns:
                df = df.withColumn(c, col("날짜") * 0)  # 빈값 생성

        df = df.select(TARGET_COLUMNS)

        # 9️⃣ DB 적재 (Spark JDBC)
        df.write.jdbc(
            url=settings.db_url,
            table="seoul_metro",
            mode="append",
            properties={
                "user": settings.db_user,
                "password": settings.db_password,
                "driver": "org.mariadb.jdbc.Driver"
            }
        )

        return {"status": True, "message": "Spark ETL 완료"}

    except Exception as e:
        return {"status": False, "error": str(e)}