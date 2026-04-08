from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  spark_host: str 
  host_ip: str
  file_dir: str
  db_url: str
  db_user: str
  db_password: str
  jar_path: str

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
  )

settings = Settings()
