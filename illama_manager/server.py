"""OpenAI-compatible API server for illama."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .model_loader import loader
from .registry import registry

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="illama-manager",
    description="OpenAI-compatible API server for Intel Arc GPUs using OpenVINO",
    version=__version__,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models (OpenAI-compatible)
# =============================================================================


class ModelObject(BaseModel):
    """OpenAI Model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "illama"


class ModelList(BaseModel):
    """OpenAI Models list response."""

    object: str = "list"
    data: list[ModelObject]


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat completion request."""

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1, le=32768)  # Increased for reasoning models
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    hide_thinking: bool = Field(default=True, description="Hide <think>...</think> content from reasoning models")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat completion response with illama extensions."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    # illama extensions (Ollama-style metrics)
    eval_count: int | None = None  # tokens generated
    prompt_eval_count: int | None = None  # tokens in prompt
    eval_duration: int | None = None  # generation time (nanoseconds)
    total_duration: int | None = None  # total request time (nanoseconds)
    tokens_per_second: float | None = None  # calculated TPS


class ProcessStatus(BaseModel):
    """Process status for /ps endpoint."""

    model: str | None
    loaded: bool
    loaded_at: float | None
    last_request: float | None
    idle_seconds: float | None


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List all available models in the registry."""
    models = registry.list_models()
    return ModelList(
        data=[
            ModelObject(
                id=m.name,
                created=int(
                    time.mktime(
                        time.strptime(m.created_at[:19], "%Y-%m-%dT%H:%M:%S")
                    )
                )
                if m.created_at
                else 0,
            )
            for m in models
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelObject:
    """Get a specific model."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return ModelObject(
        id=model.name,
        created=int(
            time.mktime(time.strptime(model.created_at[:19], "%Y-%m-%dT%H:%M:%S"))
        )
        if model.created_at
        else 0,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """Create a chat completion."""
    # Build prompt from messages
    prompt = _build_prompt(request.messages)

    try:
        if request.stream:
            return StreamingResponse(
                _stream_response(request, prompt),
                media_type="text/event-stream",
            )
        else:
            # Synchronous generation with metrics
            result = loader.generate(
                model_name=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )

            # Get the text output
            output_text = result.text

            # Filter thinking content if requested
            if request.hide_thinking:
                output_text = _filter_thinking_content(output_text)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=output_text),
                    )
                ],
                usage=Usage(
                    prompt_tokens=result.prompt_eval_count,
                    completion_tokens=result.eval_count,
                    total_tokens=result.prompt_eval_count + result.eval_count,
                ),
                # illama extensions
                eval_count=result.eval_count,
                prompt_eval_count=result.prompt_eval_count,
                eval_duration=result.eval_duration_ns,
                total_duration=result.total_duration_ns,
                tokens_per_second=round(result.tokens_per_second, 2),
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(request: ChatCompletionRequest, prompt: str):
    """Stream SSE response."""
    import json

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    try:
        for token in loader.generate(
            model_name=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        ):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Streaming failed")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _build_prompt(messages: list[ChatMessage]) -> str:
    """Build a prompt string from chat messages.

    This is a simple implementation. Production should use
    the model's actual chat template.
    """
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def _filter_thinking_content(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output.
    
    Some models (gpt-oss, DeepSeek-R1, etc.) output their reasoning in
    <think> tags before providing the actual answer.
    """
    import re
    # Remove <think>...</think> blocks (including multiline)
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also handle unclosed <think> tags (model cut off mid-thinking)
    filtered = re.sub(r'<think>.*$', '', filtered, flags=re.DOTALL)
    # Clean up extra whitespace
    filtered = re.sub(r'\n{3,}', '\n\n', filtered.strip())
    return filtered


@app.get("/ps", response_model=ProcessStatus)
async def process_status() -> ProcessStatus:
    """Show currently loaded model and stats."""
    loaded = loader.get_loaded()
    if loaded is None:
        return ProcessStatus(
            model=None,
            loaded=False,
            loaded_at=None,
            last_request=None,
            idle_seconds=None,
        )
    return ProcessStatus(
        model=loaded.info.name,
        loaded=True,
        loaded_at=loaded.loaded_at,
        last_request=loaded.last_request,
        idle_seconds=time.time() - loaded.last_request,
    )


@app.post("/unload")
async def unload_model(model: str | None = None) -> dict[str, Any]:
    """Unload the current model or a specific model."""
    success = loader.unload(model)
    return {"success": success, "model": model}


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown."""
    loader.shutdown()
