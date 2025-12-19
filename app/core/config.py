from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Backend API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # MySQL Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "pritras_db"

    # API 설정
    API_V1_PREFIX: str = "/api/v1"

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
