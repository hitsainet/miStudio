"""
Configuration settings for oLLM Server
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """oLLM Server Configuration"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 11434

    # Model settings
    default_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_cache_dir: str = "/data/ollm_models"

    # oLLM inference settings
    max_context_length: int = 8192
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use
    enable_ssd_offload: bool = True
    ssd_cache_dir: str = "/data/ollm_cache"

    # Memory management
    max_model_memory_gb: float = 20.0  # Maximum GPU memory for model loading (leave ~4GB for inference)
    inference_timeout_seconds: float = 300.0  # Timeout for model inference (5 min)
    model_load_timeout_seconds: float = 600.0  # Timeout for model loading (10 min)

    # Generation defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    default_top_p: float = 0.9

    # HuggingFace settings
    hf_token: Optional[str] = os.environ.get("HF_TOKEN")

    # CORS settings (for direct access)
    allowed_origins: list = ["http://dev-mistudio.mcslab.io", "http://localhost:3000", "http://localhost"]

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = "OLLM_"
        env_file = ".env"


settings = Settings()
