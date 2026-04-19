"""
UDMS API Configuration — Loads settings from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    MODEL_PATH: str = "models/classifier.tflite"
    LABEL_MAP_PATH: str = "models/label_map.json"
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_FILE_SIZE_MB: int = 10
    LOG_LEVEL: str = "INFO"
    MODEL_VERSION: str = "1.0.0"

    model_config = {"env_prefix": "UDMS_"}


settings = Settings()

