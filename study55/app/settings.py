from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  hf_token: str
  repo_name: str
  json_file: str
  model_qwen: str
  model_gemma: str

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
  )

settings = Settings()