import shutil
import urllib.parse
import pandas as pd
from pathlib import Path  # <--- 이 부분이 추가되었습니다!
from pyspark.sql import SparkSession
from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from sqlalchemy import create_engine
from settings import settings

app = FastAPI(title="Semicolon Subway ETL")
spark = None

# 업로드 폴더 설정
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- 1. MariaDB 연결 엔진 ---
def get_engine():
    safe_password = urllib.parse.quote_plus(settings.MARIADB_PASSWORD)
    return create_engine(
        f"mysql+pymysql://{settings.MARIADB_USER}:{safe_password}"
        f"@{settings.MARIADB_HOST}:{settings.MARIADB_PORT}/{settings.MARIADB_DB}?charset=utf8mb4"
    )

# --- 2. 라이프사이클 관리 ---
@app.on_event("startup")
def startup_event():
    global spark
    try:
        spark = SparkSession.builder \
            .appName("mySparkApp") \
            .master(settings.spark_url) \
            .config("spark.driver.host", settings.host_ip) \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .config("spark.driver.port", "10000") \
            .config("spark.blockManager.port", "10001") \
            .config("spark.cores.max", "2") \
            .getOrCreate()
        print("🚀 Spark Session Created Successfully!")
    except Exception as e:
        print(f"❌ Failed to create Spark session: {e}")

@app.on_event("shutdown")
def shutdown_event():
    global spark
    if spark:
        spark.stop()
        print("🛑 Spark Session Stopped")

# --- 3. 실제 DB 적재 함수 (데이터 유실 방지 로직 포함) ---
def process_and_insert(file_path: Path):
    try:
        engine = get_engine()
        # 1. 일단 전체를 읽습니다.
        df = pd.read_csv(str(file_path), encoding="utf-8", header=1, thousands=',', dtype=str)
        print(f"--- [DEBUG] {file_path.name} 읽기 완료 (행: {len(df)}, 열: {len(df.columns)}) ---")

        # 2. 유연한 컬럼 매핑 (이름으로 찾기)
        # 파일마다 인덱스가 달라도 '날짜', '호선', '역명'이라는 글자가 들어간 열을 찾아냅니다.
        col_map = {
            '날짜': [c for c in df.columns if '날짜' in c][0],
            '호선': [c for c in df.columns if '호선' in c][0],
            '역번호': [c for c in df.columns if '역번호' in c][0],
            '역명': [c for c in df.columns if '역명' in c][0],
        }
        
        # 시간대 컬럼들만 따로 추출 (숫자가 포함된 컬럼들)
        time_cols = [c for c in df.columns if '~' in c or ':' in c or '이전' in c or '이후' in c]
        # 합계 컬럼 찾기
        total_col = [c for c in df.columns if '합' in c and '계' in c]

        # 최종 사용할 컬럼 리스트 구성
        final_cols = [col_map['날짜'], col_map['호선'], col_map['역번호'], col_map['역명']] + time_cols + total_col
        
        # 필요한 컬럼만 필터링
        df = df[final_cols]
        
        # 우리 DB 구조에 맞게 이름 통일 (25개 컬럼)
        df.columns = [
            '날짜', '호선', '역번호', '역명', '06:00 이전', '06:00-07:00', '07:00-08:00',
            '08:00-09:00', '09:00-10:00', '10:00-11:00', '11:00-12:00', '12:00-13:00',
            '13:00-14:00', '14:00-15:00', '15:00-16:00', '16:00-17:00', '17:00-18:00',
            '18:00-19:00', '19:00-20:00', '20:00-21:00', '21:00-22:00', '22:00-23:00',
            '23:00-24:00', '24:00 이후', '승차합계'
        ]

        # 3. 날짜 보정 (이전과 동일)
        def robust_date_parser(val):
            val = str(val).strip()
            if not val or val == 'nan' or '날짜' in val: return pd.NaT
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%Y%m%d'):
                try: return pd.to_datetime(val, format=fmt)
                except: continue
            return pd.to_datetime(val, errors='coerce')

        df['날짜'] = df['날짜'].apply(robust_date_parser)
        df = df.dropna(subset=['날짜'])
        df['날짜'] = df['날짜'].dt.date

        # 4. 숫자형 변환
        num_cols = [c for c in df.columns if c not in ['날짜', '역명']]
        for col in num_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip(), 
                errors='coerce'
            ).fillna(0).astype(int)

        # 5. DB 적재
        print(f"🚀 {file_path.name} 적재 시도: {len(df)}건")
        if len(df) > 0:
            df.to_sql('승차', con=engine, if_exists='append', index=False, chunksize=2000)
            print(f"✅ 적재 완료!")
        else:
            print("❌ 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
# --- 4. 엔드포인트 ---

# (기존) 루트 경로: 미리보기 및 적재 시작
@app.get("/")
def read_root(background_tasks: BackgroundTasks):
    if not spark:
        return {"status": False, "error": "Spark session not initialized"}
    
    try:
        # 설정에 지정된 파일 경로 사용 (Path 객체로 변환)
        target_file = Path(settings.file_dir)
        
        # 미리보기 데이터 (50행)
        df_preview = pd.read_csv(str(target_file), encoding="utf-8", header=1, thousands=',', nrows=50)
        spDf = spark.createDataFrame(df_preview.astype(str))
        result = spDf.limit(50).toPandas().to_dict(orient="records")

        # 백그라운드 적재 작업 실행
        background_tasks.add_task(process_and_insert, target_file)

        return {
            "status": True, 
            "message": "데이터 미리보기 성공 및 DB 적재 시작",
            "preview_data": result
        }
    except Exception as e:
        return {"status": False, "error": str(e)}

# (추가) 업로드 경로: 새 파일 업로드 후 적재
@app.post("/upload")
async def upload_subway_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    background_tasks.add_task(process_and_insert, file_path)
    
    return {"status": "success", "message": f"{file.filename} 업로드 완료 및 적재 시작"}