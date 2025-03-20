"""
Configuration settings for ATIA Phase 4.

This module provides a centralized configuration with extended settings for Phase 4.
"""

import os
from typing import Dict, Optional, List, Any, Set

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field

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
    disable_responses_api: bool = os.getenv("DISABLE_RESPONSES_API", "False").lower() == "true"

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

    # Phase 4 additions

    # Caching settings
    enable_cache: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    max_cache_size: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))

    # Vector DB settings
    enable_vector_db: bool = os.getenv("ENABLE_VECTOR_DB", "False").lower() == "true".lower() == "true"
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "")

    # Authentication settings
    require_authentication: bool = os.getenv("REQUIRE_AUTHENTICATION", "False").lower() == "true"
    jwt_secret: str = os.getenv("JWT_SECRET", "atia_jwt_secret")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_seconds: int = int(os.getenv("JWT_EXPIRATION_SECONDS", "3600"))

    # API Gateway settings
    api_gateway_enabled: bool = os.getenv("API_GATEWAY_ENABLED", "False").lower() == "true"
    api_gateway_host: str = os.getenv("API_GATEWAY_HOST", "0.0.0.0")
    api_gateway_port: int = int(os.getenv("API_GATEWAY_PORT", "8000"))

    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE", None)
    enable_file_logging: bool = os.getenv("ENABLE_FILE_LOGGING", "False").lower() == "true"

    # Analytics settings
    enable_analytics: bool = os.getenv("ENABLE_ANALYTICS", "True").lower() == "true"
    analytics_storage_dir: str = os.getenv("ANALYTICS_STORAGE_DIR", "data/analytics")
    analytics_retention_days: int = int(os.getenv("ANALYTICS_RETENTION_DAYS", "30"))

    # Performance settings
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    parallel_requests: int = int(os.getenv("PARALLEL_REQUESTS", "5"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay_seconds: float = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))
    retry_backoff: float = float(os.getenv("RETRY_BACKOFF", "2.0"))

    # Tool Registry settings
    tool_registry_storage_dir: str = os.getenv("TOOL_REGISTRY_STORAGE_DIR", "data/tools")
    functions_storage_dir: str = os.getenv("FUNCTIONS_STORAGE_DIR", "data/functions")

    # Replace the class-based Config with model_config using ConfigDict
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    def validate_api_keys(self) -> List[str]:
        """
        Validate that required API keys are present.

        Returns:
            List of missing API keys
        """
        missing_keys = []

        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")

        return missing_keys

    def validate_settings(self) -> Dict[str, str]:
        """
        Validate all settings and return any warnings or errors.

        Returns:
            Dictionary of validation messages
        """
        validation_messages = {}

        # Check required API keys
        missing_keys = self.validate_api_keys()
        if missing_keys:
            validation_messages["missing_api_keys"] = f"Missing required API keys: {', '.join(missing_keys)}"

        # Check if both OpenAI Responses API is disabled and no OpenAI API key
        if self.disable_responses_api and not self.openai_api_key:
            validation_messages["openai_configuration"] = "Responses API is disabled, but no OpenAI API key provided"

        # Check vector DB settings
        if self.enable_vector_db and not self.pinecone_api_key:
            validation_messages["vector_db"] = "Vector DB is enabled, but no Pinecone API key provided"

        # Check analytics settings
        if self.enable_analytics and not os.path.isdir(self.analytics_storage_dir):
            try:
                os.makedirs(self.analytics_storage_dir, exist_ok=True)
            except Exception as e:
                validation_messages["analytics_storage"] = f"Failed to create analytics storage directory: {e}"

        # Check if JWT authentication is enabled without a strong secret
        if self.require_authentication and self.jwt_secret == "atia_jwt_secret":
            validation_messages["jwt_security"] = "Authentication is enabled, but using default JWT secret"

        # Check tool registry storage
        if not os.path.isdir(self.tool_registry_storage_dir):
            try:
                os.makedirs(self.tool_registry_storage_dir, exist_ok=True)
            except Exception as e:
                validation_messages["tool_registry_storage"] = f"Failed to create tool registry directory: {e}"

        # Check functions storage
        if not os.path.isdir(self.functions_storage_dir):
            try:
                os.makedirs(self.functions_storage_dir, exist_ok=True)
            except Exception as e:
                validation_messages["functions_storage"] = f"Failed to create functions directory: {e}"

        return validation_messages

    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        import logging

        # Set log level
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)

        # Basic configuration
        logging_config = {
            'level': log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }

        # Add file handler if enabled
        if self.enable_file_logging and self.log_file:
            logging_config['filename'] = self.log_file
            logging_config['filemode'] = 'a'

        # Apply configuration
        logging.basicConfig(**logging_config)

        # Set OpenAI logging level to WARNING to reduce noise
        logging.getLogger("openai").setLevel(logging.WARNING)

        # Set urllib3 logging level to WARNING to reduce noise
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# Create a global settings instance
settings = Settings()

# Automatically configure logging
settings.configure_logging()


def print_settings(include_secrets: bool = False) -> str:
    """
    Generate a printable string of current settings.

    Args:
        include_secrets: Whether to include secret values like API keys

    Returns:
        String representation of settings
    """
    # Define which fields are secrets
    secret_fields = {
        "openai_api_key", "rapid_api_key", "serp_api_key", 
        "github_token", "pinecone_api_key", "jwt_secret"
    }

    lines = ["Current Settings:"]

    # Get all settings as dict
    settings_dict = settings.model_dump()

    # Add each setting to the output
    for key, value in sorted(settings_dict.items()):
        if key in secret_fields and not include_secrets:
            # Mask the value if it's a secret and include_secrets is False
            if value:
                value = f"{'*' * 8}{value[-4:]}" if isinstance(value, str) and len(value) > 4 else "********"
            else:
                value = "Not set"

        lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def validate_environment() -> None:
    """
    Validate the environment and display warnings or errors.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Validate settings
    validation_messages = settings.validate_settings()

    if validation_messages:
        logger.warning("Environment validation found issues:")
        for category, message in validation_messages.items():
            logger.warning(f"  {category}: {message}")
    else:
        logger.info("Environment validation: All checks passed")


# Auto-validate environment when module is imported
validate_environment()