import os
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings.

    Load configuration from environment variables or .env file.
    """
    # OpenAI API configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    # OpenAI Responses API configuration
    openai_assistant_id: Optional[str] = os.getenv("OPENAI_ASSISTANT_ID", "")
    openai_responses_enabled: bool = os.getenv("OPENAI_RESPONSES_ENABLED", "False").lower() == "true"

    # RapidAPI configuration
    rapid_api_key: Optional[str] = os.getenv("RAPID_API_KEY", "")

    # SERP API (Google) configuration
    serp_api_key: Optional[str] = os.getenv("SERP_API_KEY", "")

    # GitHub API configuration
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN", "")

    # Development mode
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Tool need identifier thresholds
    need_identifier_threshold: float = float(os.getenv("NEED_IDENTIFIER_THRESHOLD", "0.75"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a global settings instance
settings = Settings()