"""Configuration management for illama-manager."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings from environment variables."""

    # Server
    host: str = field(default_factory=lambda: os.getenv("ILLAMA_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("ILLAMA_PORT", "11434")))
    log_level: str = field(default_factory=lambda: os.getenv("ILLAMA_LOG_LEVEL", "INFO"))
    debug: bool = field(default_factory=lambda: os.getenv("ILLAMA_DEBUG", "0") == "1")

    # Device
    device: str = field(default_factory=lambda: os.getenv("ILLAMA_DEVICE", "GPU"))

    # Model management
    one_model: bool = field(
        default_factory=lambda: os.getenv("ILLAMA_ONE_MODEL", "1") == "1"
    )
    idle_ttl_sec: int = field(
        default_factory=lambda: int(os.getenv("ILLAMA_IDLE_TTL_SEC", "600"))
    )

    # Paths
    registry_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ILLAMA_REGISTRY_DIR", "/opt/illama/registry")
        )
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ILLAMA_CACHE_DIR", "/opt/illama/cache"))
    )

    # HuggingFace
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    # Performance
    omp_num_threads: int = field(
        default_factory=lambda: int(os.getenv("OMP_NUM_THREADS", "8"))
    )

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
