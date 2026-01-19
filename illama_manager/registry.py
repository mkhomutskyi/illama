"""Model registry management."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model."""

    name: str
    hf_repo: str
    weight_format: str = "int4"
    family: str = "text"  # text, vlm, embedding
    size_bytes: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used: str | None = None
    tokens_per_sec: float | None = None
    gated: bool = False
    local_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """Manages the model registry on disk."""

    def __init__(self, registry_dir: Path | None = None):
        self.registry_dir = registry_dir or settings.registry_dir
        self.index_file = self.registry_dir / "index.json"
        self._models: dict[str, ModelInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.index_file.exists():
            try:
                data = json.loads(self.index_file.read_text(encoding="utf-8"))
                self._models = {
                    name: ModelInfo.from_dict(info) for name, info in data.items()
                }
                logger.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._models = {}
        else:
            self._models = {}

    def _save(self) -> None:
        """Save registry to disk."""
        data = {name: model.to_dict() for name, model in self._models.items()}
        self.index_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def list_models(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def get_model(self, name: str) -> ModelInfo | None:
        """Get a model by name."""
        return self._models.get(name)

    def add_model(self, model: ModelInfo) -> None:
        """Add or update a model in the registry."""
        self._models[model.name] = model
        self._save()
        logger.info(f"Registered model: {model.name}")

    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry."""
        if name in self._models:
            model = self._models.pop(name)
            self._save()
            # Also delete artifacts if they exist
            if model.local_path:
                model_path = Path(model.local_path)
                if model_path.exists():
                    import shutil

                    shutil.rmtree(model_path)
                    logger.info(f"Deleted model artifacts: {model_path}")
            logger.info(f"Removed model: {name}")
            return True
        return False

    def update_last_used(self, name: str) -> None:
        """Update the last used timestamp for a model."""
        if name in self._models:
            self._models[name].last_used = datetime.utcnow().isoformat()
            self._save()


# Global registry instance
registry = ModelRegistry()
