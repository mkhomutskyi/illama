"""HTTP client for communicating with illama-manager."""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests


class IllamaClient:
    """HTTP client for illama-manager API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> dict[str, Any]:
        """Check server health."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> list[dict[str, Any]]:
        """List all models in registry."""
        resp = self.session.get(f"{self.base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Get a specific model."""
        resp = self.session.get(f"{self.base_url}/v1/models/{model_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def process_status(self) -> dict[str, Any]:
        """Get process status (loaded model)."""
        resp = self.session.get(f"{self.base_url}/ps")
        resp.raise_for_status()
        return resp.json()

    def unload(self, model: str | None = None) -> dict[str, Any]:
        """Unload a model."""
        params = {"model": model} if model else {}
        resp = self.session.post(f"{self.base_url}/unload", params=params)
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Send a chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            return self._stream_chat(payload)
        else:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def chat_with_metrics(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Send a chat completion request and return full response with metrics.
        
        Returns the complete API response including:
        - choices[0].message.content
        - usage (prompt_tokens, completion_tokens, total_tokens)
        - eval_count, eval_duration, tokens_per_second, total_duration
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,  # Metrics only available in non-streaming mode
        }
        resp = self.session.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    def _stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        """Stream chat response."""
        with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get(
                            "content", ""
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
