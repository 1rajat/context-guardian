"""Model adapters for OpenAI, Anthropic, and Ollama."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Cost tables (USD per 1M tokens, input/output)
# ---------------------------------------------------------------------------
_COST_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    # Ollama — free (local)
}

# Model max context windows (tokens)
_MAX_CONTEXT: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-opus": 200_000,
    "claude-sonnet-4-5": 200_000,
}


@dataclass
class ModelResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: Optional[str] = None


class BaseAdapter(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 256) -> ModelResponse:
        ...

    def max_context(self) -> int:
        return _MAX_CONTEXT.get(self.model, 8_192)

    def cost_per_million(self) -> tuple[float, float]:
        """Returns (input_cost, output_cost) per 1M tokens in USD."""
        return _COST_TABLE.get(self.model, (0.0, 0.0))

    def estimate_cost(self, input_tokens: int, output_tokens: int = 256) -> float:
        inp, out = self.cost_per_million()
        return (input_tokens * inp + output_tokens * out) / 1_000_000


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------
class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self._client = OpenAI(api_key=key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 256) -> ModelResponse:
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            msg = resp.choices[0].message.content or ""
            usage = resp.usage
            return ModelResponse(
                content=msg,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model=self.model,
            )
        except Exception as exc:
            return ModelResponse(content="", error=str(exc), model=self.model)


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------
class AnthropicAdapter(BaseAdapter):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        # Resolve short aliases
        aliases = {
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        }
        model = aliases.get(model, model)
        super().__init__(model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 256) -> ModelResponse:
        try:
            resp = self._get_client().messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            content = resp.content[0].text if resp.content else ""
            return ModelResponse(
                content=content,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model=self.model,
            )
        except Exception as exc:
            return ModelResponse(content="", error=str(exc), model=self.model)


# ---------------------------------------------------------------------------
# Ollama adapter (local)
# ---------------------------------------------------------------------------
class OllamaAdapter(BaseAdapter):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")

    def max_context(self) -> int:
        return _MAX_CONTEXT.get(self.model, 128_000)

    def complete(self, system: str, user: str, max_tokens: int = 256) -> ModelResponse:
        try:
            import requests  # soft dependency

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            }
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "")
            return ModelResponse(
                content=content,
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                model=self.model,
            )
        except Exception as exc:
            return ModelResponse(content="", error=str(exc), model=self.model)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}
_ANTHROPIC_PREFIXES = ("claude-",)


def get_adapter(model: str, ollama_url: str = "http://localhost:11434") -> BaseAdapter:
    """Instantiate the correct adapter based on the model name."""
    if model in _OPENAI_MODELS:
        return OpenAIAdapter(model)
    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return AnthropicAdapter(model)
    # Assume Ollama for everything else
    return OllamaAdapter(model, base_url=ollama_url)


def list_known_models() -> list[str]:
    return sorted(list(_OPENAI_MODELS) + [
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus",
        "claude-3-opus-20240229",
        "claude-sonnet-4-5",
    ])
