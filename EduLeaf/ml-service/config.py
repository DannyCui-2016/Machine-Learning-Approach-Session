"""ML Service Settings"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]
    ML_HOST: str = "0.0.0.0"
    ML_PORT: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
