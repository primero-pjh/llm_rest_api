from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Backend API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database (예시)
    DATABASE_URL: str = "sqlite:///./app.db"

    # API 설정
    API_V1_PREFIX: str = "/api/v1"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
