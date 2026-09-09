"""Native Ollama HTTP API provider for SAAGA.

Communicates directly with the native Ollama daemon HTTP endpoints:
- `/api/chat` for multi-turn conversational interaction
- `/api/generate` for raw completion prompts

Provides parameter translation (e.g. `max_tokens` -> `num_predict`),
connection pooling, exponential backoff retries, and concurrent batch processing.
"""
from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from saaga.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    normalize_messages,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Provider utilizing the native Ollama REST API (`/api/chat` and `/api/generate`)."""

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 8,
        default_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the native Ollama provider.

        Args:
            model_id: Name of the Ollama model (e.g. 'llama3:8b', 'mistral').
            api_base: Base URL of the Ollama daemon. Defaults to `http://localhost:11434`.
            base_url: Alias for `api_base`.
            timeout: Request timeout in seconds (Ollama model loading may take time).
            max_retries: Number of retry attempts on network or server errors.
            retry_delay: Base delay for exponential backoff (in seconds).
            max_workers: Maximum worker threads for parallel batch requests.
            default_options: Default generation options passed to Ollama `options` dict.
        """
        super().__init__(model_id=model_id, **kwargs)

        url = api_base or base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self.api_base = self._normalize_base_url(url)

        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay
        self.max_workers = max(1, max_workers)
        self.default_options = default_options or {}

        self._session = self._create_session()

    def _normalize_base_url(self, url: str) -> str:
        """Normalize URL ensuring proper protocol and stripping any /v1 or /api paths."""
        url = url.strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"
        # If someone passed http://localhost:11434/v1 or http://localhost:11434/api, strip it
        if url.endswith("/v1"):
            url = url[:-3].rstrip("/")
        elif url.endswith("/api"):
            url = url[:-4].rstrip("/")
        return url

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2,
            max_retries=Retry(total=1, backoff_factor=0.2),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute POST request to Ollama daemon with retry logic."""
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SAAGA-OllamaProvider/1.0",
        }

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    err_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(
                        "Transient error from Ollama at %s (attempt %d/%d): %s",
                        url,
                        attempt,
                        self.max_retries,
                        err_msg,
                    )
                    last_err = RuntimeError(err_msg)
                else:
                    raise RuntimeError(
                        f"Ollama API error at {url} (status {response.status_code}): {response.text}"
                    )
            except (requests.ConnectionError, requests.Timeout) as err:
                logger.warning(
                    "Network error connecting to Ollama at %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    self.max_retries,
                    err,
                )
                last_err = err

            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
                time.sleep(sleep_time)

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts to call Ollama at {url}: {last_err}"
        ) from last_err

    def _map_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map standard generation parameters to Ollama `options` dictionary."""
        options: dict[str, Any] = dict(self.default_options)

        # Token limits
        if "max_tokens" in kwargs:
            options["num_predict"] = int(kwargs["max_tokens"])
        elif "max_new_tokens" in kwargs:
            options["num_predict"] = int(kwargs["max_new_tokens"])

        # Sampling
        if "temperature" in kwargs:
            options["temperature"] = float(kwargs["temperature"])
        if "top_p" in kwargs:
            options["top_p"] = float(kwargs["top_p"])
        if "top_k" in kwargs:
            options["top_k"] = int(kwargs["top_k"])
        if "seed" in kwargs:
            options["seed"] = int(kwargs["seed"])

        # Stop sequences
        if "stop" in kwargs:
            stop = kwargs["stop"]
            options["stop"] = [stop] if isinstance(stop, str) else list(stop)

        # Penalties
        if "repeat_penalty" in kwargs:
            options["repeat_penalty"] = float(kwargs["repeat_penalty"])
        elif "repetition_penalty" in kwargs:
            options["repeat_penalty"] = float(kwargs["repetition_penalty"])
        if "presence_penalty" in kwargs:
            options["presence_penalty"] = float(kwargs["presence_penalty"])
        if "frequency_penalty" in kwargs:
            options["frequency_penalty"] = float(kwargs["frequency_penalty"])

        # Allow passing pre-formed 'options' dict
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            options.update(kwargs["options"])

        return options

    def chat(self, messages: list[dict[str, Any] | Message], **kwargs: Any) -> LLMResponse:
        """Send chat request to Ollama `/api/chat`."""
        norm_messages = normalize_messages(messages)
        options = self._map_options(kwargs)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": norm_messages,
            "stream": False,
        }
        if options:
            payload["options"] = options

        if "format" in kwargs:
            payload["format"] = kwargs["format"]
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]

        data = self._post_with_retry("api/chat", payload)

        msg = data.get("message", {})
        text = msg.get("content", "") or ""
        finish_reason = data.get("done_reason") or ("stop" if data.get("done") else "unknown")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            raw=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def chat_batch(
        self, messages_batch: list[list[dict[str, Any] | Message]], **kwargs: Any
    ) -> list[LLMResponse]:
        """Send batch of chat requests concurrently."""
        if not messages_batch:
            return []
        if len(messages_batch) == 1:
            return [self.chat(messages_batch[0], **kwargs)]

        num_workers = min(len(messages_batch), self.max_workers)
        results: list[LLMResponse | None] = [None] * len(messages_batch)

        def _worker(idx: int, msgs: list[dict[str, Any] | Message]) -> tuple[int, LLMResponse]:
            return idx, self.chat(msgs, **kwargs)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_worker, i, msgs)
                for i, msgs in enumerate(messages_batch)
            ]
            for future in as_completed(futures):
                idx, resp = future.result()
                results[idx] = resp

        return [r for r in results if r is not None]

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send raw prompt completion to Ollama `/api/generate`."""
        options = self._map_options(kwargs)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        if "system" in kwargs:
            payload["system"] = kwargs["system"]
        if "template" in kwargs:
            payload["template"] = kwargs["template"]
        if "format" in kwargs:
            payload["format"] = kwargs["format"]
        if "raw" in kwargs:
            payload["raw"] = kwargs["raw"]
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]

        data = self._post_with_retry("api/generate", payload)

        text = data.get("response", "") or ""
        finish_reason = data.get("done_reason") or ("stop" if data.get("done") else "unknown")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            raw=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[LLMResponse]:
        """Send batch of raw prompt completions concurrently."""
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]

        num_workers = min(len(prompts), self.max_workers)
        results: list[LLMResponse | None] = [None] * len(prompts)

        def _worker(idx: int, p: str) -> tuple[int, LLMResponse]:
            return idx, self.generate(p, **kwargs)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, i, p) for i, p in enumerate(prompts)]
            for future in as_completed(futures):
                idx, resp = future.result()
                results[idx] = resp

        return [r for r in results if r is not None]
