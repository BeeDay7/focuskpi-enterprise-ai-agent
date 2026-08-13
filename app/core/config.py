from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):

    app_name: str = (
        "Enterprise AI Data Intelligence Agent"
    )

    database_url: str = (
        "sqlite:///./data/demo.db"
    )

    llm_provider: str = "demo"

    llm_api_key: str = ""

    llm_model: str = ""

    llm_base_url: str = (
        "https://api.openai.com/v1"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()