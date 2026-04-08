from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, regexp_replace, to_date, lit
from pyspark.sql.types import IntegerType
from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine, text
from settings import settings
import os
import time
import traceback

app = FastAPI()
spark = None
engine = create_engine(settings.mariadb_host)

# ==============================
# 1. Spark 시작/종료
# ==============================
@app.on_event("startup")
def startup_event():
    global spark
    try:
        spark = SparkSession.builder \
            .appName("MetroSparkApp") \
            .master(settings.spark_host) \
            .config("spark.driver.host", settings.host_ip) \
            .config("spark.network.timeout", "1200s") \
            .config("spark.rpc.message.maxSize", "512") \
            .getOrCreate()
        print("Spark Session Created Successfully!")
    except Exception as e:
        print(f"Failed to create Spark session: {e}")

@app.on_event("shutdown")
def shutdown_event():
    if spark: spark.stop()

# ==============================
# 2. Spark 정제 및 적재 함수
# ==============================
def process_and_save_sqlalchemy(pd_df, file_label):
    global spark
    
    # Pandas -> Spark 변환
    sp_df = spark.createDataFrame(pd_df)

    time_cols = [
        "05~06","06~07","07~08","08~09","09~10","10~11",
        "11~12","12~13","13~14","14~15","15~16","16~17",
        "17~18","18~19","19~20","20~21","21~22","22~23","23~24","24~"
    ]

    # Spark 정제 로직
    refined_sp_df = sp_df.withColumn("날짜", to_date(col("날짜"), "yyyy-MM-dd")) \
                         .filter(trim(col("구분")).isin(["승차", "하차"]))

    for c in time_cols:
        if c in refined_sp_df.columns:
            refined_sp_df = refined_sp_df.withColumn(
                c, regexp_replace(col(c).cast("string"), ",", "").cast(IntegerType())
            ).fillna({c: 0})
        else:
            refined_sp_df = refined_sp_df.withColumn(c, lit(0))

    # Spark -> Pandas 변환 후 SQLAlchemy 적재
    refined_pd_df = refined_sp_df.toPandas()
    refined_pd_df.to_sql(
        name="seoul_metro", 
        con=engine, 
        if_exists='append', 
        index=False, 
        chunksize=1000, 
        method='multi'
    )
    print(f"   -> {file_label}: {len(refined_pd_df):,}행 적재 완료")

# ==============================
# 3. 데이터 적재 API (Chunk 모드 적용)
# ==============================
@app.post("/load")
def load_data():
    try:
        folder_path = settings.file_dir
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and "_clean" not in f]
        
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE seoul_metro"))
            conn.commit()

        for idx, file_name in enumerate(files, 1):
            file_path = os.path.join(folder_path, file_name)
            print(f"\n[{idx}/{len(files)}] 파일 처리 시작: {file_name}")
            
            # 5만행씩 끊어서 읽기 (메모리 부족 방지)
            chunk_size = 50000 
            reader = pd.read_csv(file_path, encoding="utf-8", thousands=',', 
                                 skipinitialspace=True, chunksize=chunk_size)
            
            for i, pdf_chunk in enumerate(reader, 1):
                # 컬럼명 정리
                pdf_chunk.columns = pdf_chunk.columns.str.strip()
                if "구분.1" in pdf_chunk.columns:
                    pdf_chunk = pdf_chunk.rename(columns={"구분.1": "구분"})

                # 날짜 및 요일 처리 (에러 방지용 dropna 포함)
                if "날짜" in pdf_chunk.columns:
                    pdf_chunk["날짜"] = pd.to_datetime(pdf_chunk["날짜"], errors="coerce")
                    pdf_chunk = pdf_chunk.dropna(subset=["날짜"]) # nan 에러 해결
                    pdf_chunk["요일"] = pdf_chunk["날짜"].dt.dayofweek # 요일 분리 (0-6)
                    pdf_chunk["날짜"] = pdf_chunk["날짜"].dt.strftime("%Y-%m-%d")
                
                if not pdf_chunk.empty:
                    process_and_save_sqlalchemy(pdf_chunk, f"{file_name}_p{i}")

        return {"status": True, "message": "성공적으로 적재되었습니다."}

    except Exception as e:
        traceback.print_exc()
        return {"status": False, "error": str(e)}