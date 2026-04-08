from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark WordCount Test").master("spark://localhost:7077").getOrCreate()



# 1. rdd이용해서 csv파일,json파일 만들고 저장
# # 자바가 구동 되어야 사용 가능하기 때문에 먼저 조사
# !java -version
# # pyspark 라이브러리 설치
# !pip install pyspark

# # SparkSession 생성에 필요한 클래스 import
# from pyspark.sql import SparkSession

# # SparkContext import (Spark의 기본 실행 환경)
# from pyspark import SparkContext

# # 현재 활성화되어 있는 SparkContext 확인
# print(SparkContext._active_spark_context)

# # 만약 이미 실행 중인 SparkContext가 있으면
# if SparkContext._active_spark_context:
#     # 기존 SparkContext 종료 (충돌 방지)
#     SparkContext._active_spark_context.stop()

# # SparkSession 생성
# # appName : Spark 작업 이름
# # master("local[*]") : 현재 컴퓨터의 모든 CPU 코어를 사용하여 로컬 실행
# spark = SparkSession.builder.appName("PySpark WordCount Test").master("local[*]").getOrCreate()

# # spark 객체의 타입 확인
# print(type(spark))

# # 테스트용 문자열 리스트 (이 코드는 실제로 사용되지는 않음)
# data = ["hello spark", "hello docker", "hello worker"]

# # SparkContext를 사용하여 텍스트 파일 읽기
# # /opt/spark/works/sample.txt 파일을 읽어서 RDD 생성
# rdd = spark.sparkContext.textFile("/opt/spark/works/sample.txt")

# # WordCount 처리 과정
# counts = rdd \
#     # 문장을 공백 기준으로 분리 → 단어 리스트 생성
#     .flatMap(lambda x: x.split(" ")) \
    
#     # 각 단어를 (단어, 1) 형태로 변환
#     .map(lambda word: (word, 1)) \
    
#     # 같은 단어끼리 합산
#     .reduceByKey(lambda a, b: a + b)

# # 결과를 Python으로 가져와서 출력
# for word, count in counts.collect():
#     print(f"{word}: {count}")

# # RDD를 DataFrame으로 변환하는 코드입니다.
# df = counts.toDF(["word", "count"])

# df.show()

# # output 폴더 생성 후 내부에 csv 파일 생성
# df.coalesce(1).write.mode("overwrite").csv("/opt/spark/output/workdcount", header=True)

# # output 폴더 생성 후 내부에 json 파일 생성
# df.coalesce(1).write.mode("overwrite").json("/opt/spark/output/wordcount_json")

# # 항상 사용후 끄는 습관을 들이기 위해 하는 행위
# spark.stop()

# -------------------------------------------------------------------------

# 2. pandas와 Apache Spark(PySpark) 데이터를 서로 변환하고 SQL / RDD 처리까지 하는 예제.
# # pandas 라이브러리 설치
# !pip install pandas

# # pyspark 라이브러리 설치
# !pip install pyspark

# # pandas import
# import pandas as pd


# # 데이터 한 행씩 생성 (리스트 형태)
# row1 = ["kim", 32]
# row2 = ["park", 45]
# row3 = ["lee", 27]

# # 컬럼 이름 정의
# colNames = ["name", "age"]

# # pandas DataFrame 생성
# df1 = pd.DataFrame([row1, row2, row3], columns=colNames)

# # pandas DataFrame 출력
# print(df1)

# # SparkSession 생성 (Spark 실행 환경)
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("Spark Study").getOrCreate()


# # 리스트 데이터를 Spark DataFrame으로 생성
# # schema=colNames → 컬럼 이름 지정
# spDf = spark.createDataFrame([row1, row2, row3], schema=colNames)

# # Spark DataFrame 출력
# spDf.show()


# # pandas DataFrame을 Spark DataFrame으로 변환
# spDf2 = spark.createDataFrame(df1)

# # 결과 출력
# spDf2.show()


# # 각각의 데이터 타입 확인
# print(type(df1), type(spDf), type(spDf2))


# # Spark DataFrame → pandas DataFrame 변환
# panDf = spDf.toPandas()

# # 변환된 데이터 타입 확인
# print(type(panDf))


# # Spark DataFrame을 SQL에서 사용할 수 있도록 임시 테이블 생성
# spDf.createOrReplaceTempView("s_table")


# # DataFrame 객체 출력
# print(spDf)


# # SQL 쿼리 작성
# sql1 = """
# select name, age from s_table where age > 30
# """

# # Spark SQL 실행
# result = spark.sql(sql1)

# # 결과 출력
# result.show()


# # DataFrame API 방식 필터링
# # age가 30 이상인 데이터 조회
# result2 = spDf.filter(spDf.age >= 30)

# # 결과 출력
# result2.show()


# # SQL 표현식을 사용하기 위한 함수 import
# from pyspark.sql.functions import expr


# # 새로운 컬럼 생성
# # age가 30보다 크면 'over 30', 아니면 'lower than 30'
# result3 = spDf.withColumn(
#     "newCol",
#     expr("case when age > 30 then 'over 30' else 'lower than 30' end")
# )

# # 결과 출력
# result3.show()


# # 현재 Spark catalog에 등록된 테이블 목록 조회
# spark.catalog.listTables()


# # 임시 테이블 삭제
# spark.catalog.dropTempView("s_table")


# # Spark DataFrame → RDD 변환
# rdd = spDf.rdd

# # RDD 출력
# print(rdd)

# # RDD 타입 확인
# print(type(rdd))


# # RDD 데이터를 Python으로 가져오기
# rdd.collect()


# # DataFrame의 컬럼 목록 가져오기
# cols = spDf.columns

# # 컬럼 리스트 출력
# print(cols)


# # "name" 컬럼의 index 위치 찾기
# nameCol = cols.index("name")

# print(nameCol)


# # RDD 필터링
# # age가 27 이상인 데이터 추출
# # (주의: 코드에 ageCol 변수가 정의되지 않아서 실제 실행 시 오류 가능)
# result4 = rdd.filter(lambda x: x[ageCol] >= 27)

# # 결과 출력
# result4.collect()


# # 새로운 컬럼을 만드는 함수 정의
# def newCol(x):
    
#     # 기존 이름
#     oName = x.name
    
#     # 기존 나이
#     oAge = x.age
    
#     # 이름 뒤에 "님" 추가
#     name = oName + "님"
    
#     # 나이를 10배로 증가
#     age = oAge * 10
    
#     # 새로운 문자열 컬럼 생성
#     newCol = f"{name} {age}"
    
#     # 튜플 형태로 반환
#     return(name, age, newCol)


# # map을 이용해 RDD 데이터 변환
# result5 = rdd.map(lambda x: newCol(x))

# # 결과 출력
# result5.collect()


# # Spark 세션 종료
# spark.stop()


# 2번 코드에서 중요한 Spark 기능
# 기능	                     설명
# createDataFrame	        데이터 → Spark DataFrame
# toPandas	                Spark → pandas
# createOrReplaceTempView	SQL 테이블 생성
# spark.sql	                SQL 실행
# filter	                DataFrame 조건 검색
# withColumn	            새로운 컬럼 생성
# rdd	                    DataFrame → RDD

spDf.createOrReplaceTempView("s_table")

sql1 = """
select name, age from s_table where age > 30
"""

fldDf =spark.sql(sql1)
fldDf.show()

spark.stop()