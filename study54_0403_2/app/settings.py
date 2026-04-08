from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  input_dir: str = "books"
  target_dir: str = "datasets"
  model_dir: str = "models"
  prefix_name: str = "cleaned_"
  save_dir: str = "export"

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
  )

settings = Settings()