"""Base interfaces and data structures for SAAGA LLM providers.

Defines the universal contracts:
- `Message`: Standardized chat message representation.
- `LLMResponse`: Standardized completion/chat generation result.
- `BaseLLMProvider`: Abstract base class for all LLM backends (OpenAI-compatible,
  Ollama, vLLM, HuggingFace, Mock).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message with role and text content."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a standard dictionary format."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        """Create a Message from a dictionary."""
        return cls(role=str(d.get("role", "user")), content=str(d.get("content", "")))


@dataclass
class LLMResponse:
    """Standardized response container for all LLM calls."""

    text: str
    raw: dict[str, Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        """Total tokens used across prompt and completion."""
        return self.prompt_tokens + self.completion_tokens


def normalize_messages(messages: Sequence[dict[str, Any] | Message]) -> list[dict[str, str]]:
    """Normalize a sequence of dicts or Message instances into list of dicts."""
    normalized: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, Message):
            normalized.append(m.to_dict())
        elif isinstance(m, dict):
            normalized.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
        else:
            raise TypeError(f"Expected dict or Message, got {type(m).__name__}: {m!r}")
    return normalized


def messages_to_prompt(messages: Sequence[dict[str, Any] | Message]) -> str:
    """Fallback plain-text prompt conversion when chat templates are unavailable."""
    norm = normalize_messages(messages)
    lines: list[str] = []
    for m in norm:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM backends in SAAGA.

    Subclasses must implement `chat` at minimum. Default implementations
    are provided for `chat_batch`, `generate`, and `generate_batch` using
    sensible delegation, but providers may override them with native
    batching or specialized completions.
    """

    def __init__(self, model_id: str = "", **kwargs: Any) -> None:
        self.model_id = model_id

    @abstractmethod
    def chat(self, messages: list[dict[str, Any] | Message], **kwargs: Any) -> LLMResponse:
        """Send a single multi-turn chat request to the LLM.

        Args:
            messages: List of message dicts or `Message` objects.
            **kwargs: Generation parameters (e.g. temperature, max_tokens, stop).

        Returns:
            LLMResponse containing generated text and usage metadata.
        """
        ...

    def chat_batch(
        self, messages_batch: list[list[dict[str, Any] | Message]], **kwargs: Any
    ) -> list[LLMResponse]:
        """Send a batch of multi-turn chat requests.

        Default implementation processes requests sequentially. Subclasses
        should override this with concurrent or engine-native batching.
        """
        return [self.chat(messages, **kwargs) for messages in messages_batch]

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send a single raw text completion prompt to the LLM.

        Default implementation wraps the prompt as a user chat message.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[LLMResponse]:
        """Send a batch of raw text completion prompts.

        Default implementation wraps each prompt as a user chat message
        and delegates to `chat_batch`.
        """
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        return self.chat_batch(messages_batch, **kwargs)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for unit tests, offline development, and verification.

    Allows providing canned responses, sequential cycles, or custom callbacks.
    Records all invocation history for test assertions.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        responses: list[str] | None = None,
        default_response: str = "mock response",
        callback: Callable[[list[dict[str, str]], dict[str, Any]], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.canned_responses = list(responses) if responses is not None else []
        self.default_response = default_response
        self.callback = callback
        self.history: list[dict[str, Any]] = []

    def _next_text(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        if self.callback is not None:
            return self.callback(messages, kwargs)
        if self.canned_responses:
            return self.canned_responses.pop(0)
        return self.default_response

    def chat(self, messages: list[dict[str, Any] | Message], **kwargs: Any) -> LLMResponse:
        norm = normalize_messages(messages)
        text = self._next_text(norm, kwargs)
        record = {
            "type": "chat",
            "messages": norm,
            "kwargs": kwargs,
            "response": text,
        }
        self.history.append(record)
        return LLMResponse(
            text=text,
            raw={"mock": True, "record": record},
            prompt_tokens=sum(len(m["content"].split()) for m in norm),
            completion_tokens=len(text.split()),
            finish_reason="stop",
        )

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        norm = [{"role": "user", "content": prompt}]
        text = self._next_text(norm, kwargs)
        record = {
            "type": "generate",
            "prompt": prompt,
            "kwargs": kwargs,
            "response": text,
        }
        self.history.append(record)
        return LLMResponse(
            text=text,
            raw={"mock": True, "record": record},
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
        )
