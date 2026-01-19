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


@dataclass
class GenerationResult:
    """Result of text generation with performance metrics."""

    text: str
    eval_count: int  # tokens generated
    prompt_eval_count: int  # tokens in prompt (estimated)
    eval_duration_ns: int  # generation time in nanoseconds
    total_duration_ns: int  # total request time in nanoseconds
    tokens_per_second: float  # calculated TPS

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_duration": self.eval_duration_ns,
            "total_duration": self.total_duration_ns,
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


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
    ) -> GenerationResult | Iterator[str]:
        """Generate text using the specified model.
        
        Returns:
            GenerationResult with metrics for non-streaming mode
            Iterator[str] for streaming mode (metrics not available during stream)
        """
        start_time = time.time_ns()
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
            # Non-streaming with metrics
            gen_start_time = time.time_ns()
            output = loaded.pipe.generate(prompt, config)
            gen_end_time = time.time_ns()
            
            # Calculate metrics
            eval_duration_ns = gen_end_time - gen_start_time
            total_duration_ns = gen_end_time - start_time
            
            # Estimate token counts (simple word-based approximation)
            # TODO: Use tokenizer for accurate count if available
            prompt_tokens = len(prompt.split())
            output_tokens = len(output.split())
            
            # Try to get perf metrics from OpenVINO if available
            try:
                perf_metrics = loaded.pipe.get_perf_metrics()
                if hasattr(perf_metrics, 'get_num_generated_tokens'):
                    output_tokens = perf_metrics.get_num_generated_tokens()
                if hasattr(perf_metrics, 'get_num_input_tokens'):
                    prompt_tokens = perf_metrics.get_num_input_tokens()
            except Exception:
                pass  # Use word-based estimation
            
            # Calculate tokens per second
            eval_duration_sec = eval_duration_ns / 1e9
            tps = output_tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0
            
            return GenerationResult(
                text=output,
                eval_count=output_tokens,
                prompt_eval_count=prompt_tokens,
                eval_duration_ns=eval_duration_ns,
                total_duration_ns=total_duration_ns,
                tokens_per_second=tps,
            )

    def _stream_generate(
        self, loaded: LoadedModel, prompt: str, config: Any
    ) -> Iterator[str]:
        """Stream generation token by token using OpenVINO GenAI streamer."""
        import queue
        import threading
        
        # Use a queue to pass tokens from callback to iterator
        token_queue: queue.Queue[str | None] = queue.Queue()
        
        # Define streamer callback function
        def streamer_callback(token: str) -> bool:
            """Called for each generated token. Return False to stop generation."""
            token_queue.put(token)
            return False  # Continue generation
        
        # Run generation in a thread so we can yield tokens
        def run_generation():
            try:
                loaded.pipe.generate(prompt, config, streamer_callback)
            except Exception as e:
                logger.exception("Generation error in stream")
            finally:
                token_queue.put(None)  # Signal completion
        
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()
        
        # Yield tokens as they arrive
        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._stop_idle_checker.set()
        with self._lock:
            self._unload_current()


# Global loader instance
loader = ModelLoader()
