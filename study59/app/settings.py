from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  ollama_base_url: str
  graph_image_path: str = "images"

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
  )

settings = Settings()