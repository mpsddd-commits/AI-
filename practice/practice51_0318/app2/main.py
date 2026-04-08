from pyspark.sql import SparkSession
from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine, text
from settings import settings
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
# 조회 API (기존 유지)
# ==============================
@app.get("/")
def read_root():
    if not spark:
        return {"status": False, "error": "Spark session not initialized"}
    try:
        df = pd.read_csv(
            settings.file_dir,
            encoding="utf-8",
            header=0,
            thousands=',',
            quotechar='"',
            skipinitialspace=True
        )
        spDf = spark.createDataFrame(df)
        result = spDf.limit(50).toPandas().to_dict(orient="records")
        return {"status": True, "data": result}
    except Exception as e:
        return {"status": False, "error": str(e)}


# ==============================
# 데이터 정제 로직
# ==============================
TARGET_COLUMNS = [
    "날짜", "요일", "역번호", "구분",
    "05~06","06~07","07~08","08~09","09~10",
    "10~11","11~12","12~13","13~14","14~15",
    "15~16","16~17","17~18","18~19","19~20",
    "20~21","21~22","22~23","23~24","24~"
]

VALID_TYPES = ["승차", "하차"]


def normalize_columns(df):
    new_cols = {}
    for col in df.columns:
        c = col.strip()
        c = c.replace(" ", "").replace(":", "").replace("시", "").replace("-", "~")

        if "이전" in c:
            new_cols[col] = "05~06"
        elif "이후" in c or c == "00~01":
            new_cols[col] = "24~"
        else:
            new_cols[col] = c

    return df.rename(columns=new_cols)


def normalize_csv(file_path):
    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        thousands=",",
        quotechar='"',
        skipinitialspace=True
    )

    df.columns = df.columns.str.strip()
    df = normalize_columns(df)

    if "구분.1" in df.columns:
        df = df.drop(columns=["구분"], errors="ignore")
        df = df.rename(columns={"구분.1": "구분"})

    df = df.drop(columns=["연번", "호선", "역명", "합계"], errors="ignore")

    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df = df.dropna(subset=["날짜"])
        df["요일"] = df["날짜"].dt.dayofweek
        df["날짜"] = df["날짜"].dt.strftime("%Y-%m-%d")

    if "구분" in df.columns:
        df["구분"] = df["구분"].astype(str).str.strip()
        df = df[df["구분"].isin(VALID_TYPES)]

    for col in df.columns:
        if "~" in col:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    for col in TARGET_COLUMNS:
        if "~" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df[TARGET_COLUMNS]

    temp_file = file_path.replace(".csv", "_clean.csv")
    df.to_csv(temp_file, index=False, encoding="utf-8")

    return temp_file


# ==============================
# DB 적재
# ==============================
def load_to_db(conn, file_path):
    safe_path = file_path.replace("\\", "\\\\")

    sql = f"""
    LOAD DATA LOCAL INFILE '{safe_path}'
    INTO TABLE db_metro.seoul_metro
    CHARACTER SET utf8
    FIELDS TERMINATED BY ','
    OPTIONALLY ENCLOSED BY '"'
    LINES TERMINATED BY '\\n'
    IGNORE 1 LINES
    (
        @날짜, @요일, @역번호, @역명, @구분,
        @v05,@v06,@v07,@v08,@v09,
        @v10,@v11,@v12,@v13,@v14,
        @v15,@v16,@v17,@v18,@v19,
        @v20,@v21,@v22,@v23,@v24
    )
    SET
        날짜 = NULLIF(STR_TO_DATE(@날짜, '%Y-%m-%d'), '0000-00-00'),
        요일 = NULLIF(@요일, ''),
        역번호 = NULLIF(@역번호, ''),
        역명 = NULLIF(@역명, ''),
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
@app.post("/load")
def load_data():
    try:
        engine = create_engine(
            settings.mariadb_host,
            connect_args={"local_infile": 1}
        )

        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE db_metro.seoul_metro2"))

            folder_path = settings.file_dir

            for file in os.listdir(folder_path):
                if file.endswith(".csv") and "_clean" not in file:
                    file_path = os.path.join(folder_path, file)

                    try:
                        print(f"처리중: {file}")
                        clean_file = normalize_csv(file_path)
                        load_to_db(conn, clean_file)
                    except Exception as e:
                        print(f"실패: {file} / {e}")

            conn.commit()

        return {"status": True, "message": "데이터 적재 완료"}

    except Exception as e:
        return {"status": False, "error": str(e)}