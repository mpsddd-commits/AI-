import os

class Settings:
    ENV = os.getenv("ENV", "docker")

    if ENV == "docker":
        file_dir = "/data"
        spark_host = "spark://spark-master:7077"
        mariadb_host = "mysql+pymysql://root:1234@mariadb2:23306/db_metro"

settings = Settings()