from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- NPI registry defaults ---
    NPI_BASE_URL: str = "https://npiregistry.cms.hhs.gov/api/"
    NPI_API_VERSION: str = "2.1"

    # --- HTTP timeouts ---
    HTTP_CONNECT_TIMEOUT: float = 10.0
    HTTP_READ_TIMEOUT: float = 20.0
    HTTP_WRITE_TIMEOUT: float = 10.0
    HTTP_POOL_TIMEOUT: float = 5.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()