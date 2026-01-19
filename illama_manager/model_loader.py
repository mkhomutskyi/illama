"""OpenVINO model loading and inference."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import settings
from .registry import ModelInfo, registry

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Represents a currently loaded model."""

    info: ModelInfo
    pipe: Any  # OpenVINO GenAI pipeline
    loaded_at: float
    last_request: float


class ModelLoader:
    """Manages model loading, unloading, and inference.

    Enforces single-model-at-a-time policy when ILLAMA_ONE_MODEL=1.
    """

    def __init__(self):
        self._current_model: LoadedModel | None = None
        self._lock = threading.Lock()
        self._idle_checker: threading.Thread | None = None
        self._stop_idle_checker = threading.Event()

        # Start idle checker thread
        if settings.idle_ttl_sec > 0:
            self._start_idle_checker()

    def _start_idle_checker(self) -> None:
        """Start background thread to unload idle models."""

        def check_idle():
            while not self._stop_idle_checker.wait(timeout=30):
                self._check_and_unload_idle()

        self._idle_checker = threading.Thread(target=check_idle, daemon=True)
        self._idle_checker.start()

    def _check_and_unload_idle(self) -> None:
        """Unload model if idle beyond TTL."""
        with self._lock:
            if self._current_model is None:
                return
            idle_time = time.time() - self._current_model.last_request
            if idle_time > settings.idle_ttl_sec:
                logger.info(
                    f"Unloading idle model: {self._current_model.info.name} "
                    f"(idle for {idle_time:.0f}s)"
                )
                self._unload_current()

    def _unload_current(self) -> None:
        """Unload the current model. Must hold lock."""
        if self._current_model is not None:
            model_name = self._current_model.info.name
            # Release OpenVINO resources
            self._current_model.pipe = None
            self._current_model = None
            logger.info(f"Unloaded model: {model_name}")

    def _load_model(self, model_info: ModelInfo) -> LoadedModel:
        """Load a model into memory. Must hold lock."""
        logger.info(f"Loading model: {model_info.name} from {model_info.local_path}")

        if not model_info.local_path:
            raise ValueError(f"Model {model_info.name} has no local path")

        model_path = Path(model_info.local_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Import OpenVINO GenAI (lazy import to avoid startup cost)
        try:
            import openvino_genai as ov_genai
        except ImportError as e:
            raise ImportError(
                "openvino-genai not installed. Install with: pip install openvino-genai"
            ) from e

        # Create the pipeline
        pipe = ov_genai.LLMPipeline(str(model_path), settings.device)

        now = time.time()
        loaded = LoadedModel(
            info=model_info,
            pipe=pipe,
            loaded_at=now,
            last_request=now,
        )

        logger.info(f"Model loaded: {model_info.name}")
        return loaded

    def ensure_loaded(self, model_name: str) -> LoadedModel:
        """Ensure a model is loaded, loading it if necessary.

        If one_model mode is enabled, unloads any other model first.
        """
        with self._lock:
            # Already loaded?
            if self._current_model and self._current_model.info.name == model_name:
                self._current_model.last_request = time.time()
                return self._current_model

            # Get model info from registry
            model_info = registry.get_model(model_name)
            if model_info is None:
                raise ValueError(f"Model not found in registry: {model_name}")

            # Unload current if one_model mode
            if settings.one_model and self._current_model:
                logger.info(
                    f"Switching model: {self._current_model.info.name} -> {model_name}"
                )
                self._unload_current()

            # Load the new model
            self._current_model = self._load_model(model_info)
            registry.update_last_used(model_name)
            return self._current_model

    def get_loaded(self) -> LoadedModel | None:
        """Get the currently loaded model, if any."""
        with self._lock:
            return self._current_model

    def unload(self, model_name: str | None = None) -> bool:
        """Unload a model by name, or the current model if no name given."""
        with self._lock:
            if self._current_model is None:
                return False
            if model_name and self._current_model.info.name != model_name:
                return False
            self._unload_current()
            return True

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate text using the specified model."""
        loaded = self.ensure_loaded(model_name)

        with self._lock:
            loaded.last_request = time.time()

        # Configure generation
        config = loaded.pipe.get_generation_config()
        config.max_new_tokens = max_tokens
        config.temperature = temperature

        if stream:
            return self._stream_generate(loaded, prompt, config)
        else:
            return loaded.pipe.generate(prompt, config)

    def _stream_generate(
        self, loaded: LoadedModel, prompt: str, config: Any
    ) -> Iterator[str]:
        """Stream generation token by token."""
        # OpenVINO GenAI streaming interface
        streamer = loaded.pipe.generate(prompt, config, streamer=True)
        for token in streamer:
            yield token

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._stop_idle_checker.set()
        with self._lock:
            self._unload_current()


# Global loader instance
loader = ModelLoader()
