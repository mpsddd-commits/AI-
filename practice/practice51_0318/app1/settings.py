from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  spark_url: str
  host_ip: str
  file_dir: str
  mariadb_host: str
  db_url: str
  jar_path: str


  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
  )

settings = Settings()
