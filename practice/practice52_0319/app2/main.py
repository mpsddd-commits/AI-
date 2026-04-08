from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine, text
from settings import settings
from pathlib import Path
import os
import shutil

app = FastAPI()

spark = None

# ==============================
# Spark 시작/종료
# ==============================
@app.on_event("startup")
def startup_event():
    global spark
    try:
        spark = SparkSession.builder \
            .appName("mySparkApp") \
            .master(settings.spark_host) \
            .config("spark.driver.host", settings.host_ip) \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .config("spark.driver.port", "10000") \
            .config("spark.blockManager.port", "10001") \
            .config("spark.cores.max", "2") \
            .getOrCreate()
        print("Spark Session Created Successfully!")
    except Exception as e:
        print(f"Failed to create Spark session: {e}")


@app.on_event("shutdown")
def shutdown_event():
    if spark:
        spark.stop()


# ==============================
# Spark 데이터 정제 로직
# ==============================
TARGET_COLUMNS = [
    "날짜", "요일", "역번호", "구분",
    "05~06","06~07","07~08","08~09","09~10",
    "10~11","11~12","12~13","13~14","14~15",
    "15~16","16~17","17~18","18~19","19~20",
    "20~21","21~22","22~23","23~24","24~"
]

def clean_with_spark(file_path):
    global spark
    
    # 1. 파일명만 추출 (예: "D:/IDE/work/2008.csv" -> "2008.csv")
    file_name = os.path.basename(file_path)
    
    # 2. 컨테이너 내부의 절대 경로로 강제 지정
    # file:/// 를 빼고 그냥 리눅스 경로인 /data/파일명.csv 로 만듭니다.
    spark_path = f"file:///data/{file_name}"

    print(f"DEBUG: Spark 워커가 읽으려는 실제 경로 -> {spark_path}")

    # 3. 파일 읽기
    try:
        # 워커는 이제 "/data/2008.csv"를 정확히 찾아낼 겁니다.
        df = spark.read.csv(spark_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"CSV 읽기 실패: {e}")
        raise e # 상위 load_data에서 잡도록 던짐
    

    # 1. 컬럼명 정규화
    for old_col in df.columns:
        new_col = old_col.strip().replace(" ", "").replace(":", "").replace("시", "").replace("-", "~")
        if "이전" in new_col: new_col = "05~06"
        elif "이후" in new_col or new_col == "00~01": new_col = "24~"
        df = df.withColumnRenamed(old_col, new_col)

    # 2. '구분' 컬럼 처리
    if "구분.1" in df.columns:
        df = df.drop("구분").withColumnRenamed("구분.1", "구분")
    
    # [수정] DB SQL 구조에 맞게 역명 컬럼 유지 또는 추가
    if "역명" not in df.columns:
        df = df.withColumn("역명", F.lit("알수없음"))

    # 3. 데이터 타입 및 필터링
    df = df.filter(F.col("구분").isin(["승차", "하차"]))
    
    # 4. 컬럼 순서 및 타입 강제 (SQL 로직과 일치시킴)
    # TARGET_COLUMNS를 SQL 순서에 맞게 재정의하는 것이 좋습니다.
    final_cols = []
    for col in TARGET_COLUMNS:
        if col == "날짜":
            df = df.withColumn("날짜", F.to_date(F.col("날짜")))
            final_cols.append("날짜")
        elif col == "요일":
            # 월(0)~일(6) 기준으로 맞추려면: (dayofweek + 5) % 7
            df = df.withColumn("요일", (F.dayofweek(F.col("날짜")) + 5) % 7)
            final_cols.append("요일")
        elif "~" in col or col == "24~":
            clean_val = F.regexp_replace(F.col(col).cast("string"), ",", "")
            df = df.withColumn(col, clean_val.cast("int").fillna(0))
            final_cols.append(col)
        else:
            if col not in df.columns:
                df = df.withColumn(col, F.lit(None))
            final_cols.append(col)

    return df.select(*final_cols)

def load_to_db(conn, actual_csv_path):
    # 실제 파일 경로를 MariaDB 형식에 맞게 이스케이프
    safe_path = actual_csv_path.replace("\\", "\\\\")

    sql = f"""
    LOAD DATA LOCAL INFILE '{safe_path}'
    INTO TABLE db_metro.seoul_metro
    CHARACTER SET utf8
    FIELDS TERMINATED BY ','
    OPTIONALLY ENCLOSED BY '"'
    LINES TERMINATED BY '\\n'
    IGNORE 1 LINES
    (
        @날짜, @요일, @역번호, @구분,
        @v05,@v06,@v07,@v08,@v09,
        @v10,@v11,@v12,@v13,@v14,
        @v15,@v16,@v17,@v18,@v19,
        @v20,@v21,@v22,@v23,@v24
    )
    SET
        날짜 = NULLIF(STR_TO_DATE(@날짜, '%Y-%m-%d'), '0000-00-00'),
        요일 = NULLIF(@요일, ''),
        역번호 = NULLIF(@역번호, ''),
        구분 = NULLIF(@구분, ''),
        `05~06`=@v05, `06~07`=@v06, `07~08`=@v07,
        `08~09`=@v08, `09~10`=@v09, `10~11`=@v10,
        `11~12`=@v11, `12~13`=@v12, `13~14`=@v13,
        `14~15`=@v14, `15~16`=@v15, `16~17`=@v16,
        `17~18`=@v17, `18~19`=@v18, `19~20`=@v19,
        `20~21`=@v20, `21~22`=@v21, `22~23`=@v22,
        `23~24`=@v23, `24~`=@v24;
    """
    conn.execute(text(sql))

# ==============================
# 적재 API (핵심 추가)
# ==============================
def process_and_load_with_spark(file_path, conn):
    file_name = os.path.basename(file_path)
    # Spark가 저장할 때는 컨테이너 기준 경로 사용
    spark_temp_output = f"/data/{file_name.replace('.csv', '_spark_temp')}"
    
    # 호스트(데스크탑)에서 os.path로 접근할 때는 기존 방식 유지
    local_temp_output_dir = file_path.replace(".csv", "_spark_temp")
    
    try:
        # 1. 정제 수행
        df_cleaned = clean_with_spark(file_path)
        
        # 워커에게 컨테이너 내부의 /data/... 경로에 쓰라고 명령
        df_cleaned.coalesce(1).write.csv(
            spark_temp_output, 
            header=True, 
            mode="overwrite", 
            quoteAll=True,
            encoding="utf-8"
        )
        
        # 이후 로직(shutil, os.listdir 등)은 데스크탑 파이썬이 수행하므로 
        # local_temp_output_dir를 사용하면 됩니다.
        
        # 2. 파일 찾기
        actual_csv = None
        if os.path.exists(local_temp_output_dir):
            for f in os.listdir(local_temp_output_dir):
                if f.endswith(".csv") and not f.startswith("."): # 임시파일(.) 제외
                    actual_csv = os.path.join(local_temp_output_dir, f)
                    break
        
        if actual_csv:
            conn.execute(text("SET NAMES utf8mb4"))
            load_to_db(conn, actual_csv)
            print(f"Successfully loaded: {actual_csv}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        raise e # 상위 load_data에서 잡도록 던짐
    finally:
        # [핵심] DB 적재 후 잠시 대기하거나 확실하게 삭제 시도
        import time
        time.sleep(1) # Spark가 파일 핸들을 놓을 시간을 아주 잠깐 줌
        if os.path.exists(local_temp_output_dir):
            shutil.rmtree(local_temp_output_dir, ignore_errors=True)
            print(f"Temporary directory {local_temp_output_dir} cleaned.")

@app.post("/load")
def load_data():
    try:
        # SQLAlchemy 엔진 설정 (LOCAL INFILE 허용)
        engine = create_engine(
            settings.mariadb_host,
            connect_args={"local_infile": 1}
        )

        with engine.connect() as conn:
            # 기존 테이블 비우기
            conn.execute(text("TRUNCATE TABLE db_metro.seoul_metro"))

            folder_path = settings.file_dir
            for file in os.listdir(folder_path):
                if file.endswith(".csv") and "_temp" not in file:
                    file_path = os.path.join(folder_path, file)
                    process_and_load_with_spark(file_path, conn)
            
            conn.commit()
        return {"status": True, "message": "Spark 정제 및 Local Infile 적재 완료"}

    except Exception as e:
        return {"status": False, "error": str(e)}