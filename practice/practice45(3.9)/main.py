from fastapi import FastAPI
from settings import settings
import mariadb

conn_params = {
  "user" : settings.mariadb_user,
  "password" : settings.mariadb_password,
  "host" : settings.mariadb_host,
  "database" : settings.mariadb_database,
  "port" : settings.mariadb_port
}

def getConn():
  try:
    conn = mariadb.connect(**conn_params)
    if conn == None:
      return None
    return conn
  except mariadb.Error as e:
    print(f"접속 오류 : {e}")
    return None

app = FastAPI()

@app.get("/")
def read_root():
  return {"Hello": "World"}

def etl(year:int, month: int):
    print("db_air에서 db_to_air `비행`데이터 이관 작업")
    result = False
    try:
        conn = getConn()
        if conn:
            where = f"where 년도 = {year} and 월 = {month}"
            sql1 = f"""
                delete from db_to_air.`비행` {where} ;                
            """
            sql2 = f"""
                insert into db_to_air.`비행`
                SELECT * from db_air.`비행` {where};             
            """
            sql3 = f"""
                SELECT count(*) as cnt from db_to_air.`비행` {where};             
            """
            cur = conn.cursor()
            cur.execute(sql1) 
            cur.execute(sql2)
            conn.commit()
            cur.execute(sql3)
            cnt = cur.fetchone()
            print(f"적재 갯수 : {cnt[0]} 개")                     
            cur.close()
            conn.close()
            result = True
    except mariadb.Error as e:
        print(f"MariaDB Error : {e}")
    return result

def etl2(table: str, year: int = 0, month: int = 0):
    print("db_air에서 db_to_air 데이터 이관 작업")
    try:
        conn = mariadb.connect(**conn_params)
        if conn:
            where =""
            if year > 0 and month > 0:
                where = f"where 년도 = {year} and 월 = {month}"
            sql1 = f"""
                delete from db_to_air.`{table}` {where};               
            """
            sql2 = f"""
                insert into db_to_air.`{table}`
                SELECT * from db_air.`{table}` {where};             
            """
            sql3 = f"""
                SELECT count(*) as cnt from db_to_air.`{table}` {where};             
            """
            print("SQL 실행")
            cur = conn.cursor()
            cur.execute(sql1) 
            cur.execute(sql2)
            conn.commit()
            cur.execute(sql3)
            cnt = cur.fetchone()
            print(f"{table} 적재 갯수 : {cnt[0]} 개")                     
            cur.close()
            conn.close()
    except mariadb.Error as e:
        print(f"MariaDB Error : {e}")

@app.post("/set")
def useYn(table: str,useYn : bool):
    print("useYn 값 변동")
    try:
        conn = mariadb.connect(**conn_params)
        if conn:
            sql = f"update db_to_air.`jobs` set useYn = {useYn} where `table` = '{table}' "
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
            conn.close()
            result = f"{table}의 useYn의 값이 {useYn}값으로 수정 되었습니다."
            print(f"{table}의 useYn의 값이 {useYn}값으로 수정 되었습니다.")
            return {"Status": True, "result": result}
    except mariadb.Error as e:
        print(f"MariaDB Error : {e}")

@app.post("/run")
def run():
    useYn = (1,)
    job_list = jobs(useYn)
    for row in job_list:
        etl3(row)
    return {"status": True, "message": f"{len(job_list)}개의 작업이 완료되었습니다."}

@app.post("/list")
def list():
    result = []
    try:
        conn = getConn()
        if conn:
            cur = conn.cursor()
            sql =f"""
            select * from db_to_air.jobs;
            """
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            cur.close()
            conn.close()
            result = [dict(zip(columns, row)) for row in rows]
    except mariadb.Error as e:
        print(f"MariaDB Error : {e}")
    return {"status" : True, "result" : result}

def etl3(data : dict):
    print("db_air에서 db_to_air 데이터 이관 작업")
    try:
        conn = mariadb.connect(**conn_params)
        if conn:
            no = data["no"]
            year = data["year"]
            month = data["month"]
            table = data["table"]
            where =""
            if year > 0 and month > 0:
                where = f"where 년도 = {year} and 월 = {month}"
            sql1 = f"""
                delete from db_to_air.`{table}` {where};               
            """
            sql2 = f"""
                insert into db_to_air.`{table}` SELECT * from db_air.`{table}` {where};             
            """
            sql3 = f"""
                SELECT count(*) as cnt from db_to_air.`{table}` {where};             
            """
            print("SQL 실행")
            cur = conn.cursor()
            cur.execute(sql1) 
            cur.execute(sql2)
            conn.commit()
            cur.execute(sql3)
            cnt = cur.fetchone()
            print(f"{table} 적재 : {cnt[0]} 건")
            sql4 = f"update db_to_air.jobs set `cnt` = {cnt[0]}, `modDate` = now() where `no` = {no}"   
            cur.execute(sql4)   
            conn.commit()
            cur.close()
            conn.close()
    except mariadb.Error as e:
        print(f"MariaDB Error : {e}")

def jobs(useYn: tuple):
    try:
        conn = mariadb.connect(**conn_params)
        if conn:
            if isinstance(useYn, (list, tuple)):
                keys =",".join(map(str, useYn))
            else:
                keys = useYn
            sql = f"select `no`,`table`,`year`,`month` from db_to_air.jobs where useYn in ({keys})"
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            cur.close()
            conn.close()
            result = [dict(zip(columns, row)) for row in rows]
            return result
    except mariadb.Error as e:
        print(f"MariaDB Error : {e}")      
    return []  
    
if __name__ == "__main__":
    useYn = tuple([1])
    for row in jobs(useYn):
        if row: etl3(row)
        # etl2(row["table"], row["year"], row["month"])
    # etl2("비행", 1987, 10)
    # etl2("운반대")
    # etl2("항공사")

# @app.post("/run")
# def run():
#     try:
#         conn = mariadb.connect(**conn_params)
#         if conn:
#             sql = f"update db_to_air.`jobs` set useYn = {useYn} where `table` = '{table}' "
#             print("SQL 실행")
#             cur = conn.cursor()
#             cur.execute(sql)
#             conn.commit()
#             cur.close()
#             conn.close()
#     except mariadb.Error as e:
#         print(f"MariaDB Error : {e}")
