"""OpenAI-compatible LLM provider for SAAGA.

Supports any backend implementing the OpenAI Chat/Completions REST specification:
- Locally hosted vLLM server (`http://localhost:8000/v1`)
- Ollama OpenAI compatibility endpoint (`http://localhost:11434/v1`)
- Cloud providers: OpenAI, DeepSeek, Together, Groq, OpenRouter, etc.

Uses `requests` with persistent sessions, HTTP connection pooling,
exponential backoff retries, and thread-pool batch execution.
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


class OpenAIProvider(BaseLLMProvider):
    """Provider communicating with any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 16,
        default_headers: dict[str, str] | None = None,
        use_legacy_completions: bool = False,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI-compatible provider.

        Args:
            model_id: Target model identifier on the backend server.
            api_base: Base URL of the OpenAI-compatible API (e.g. 'http://localhost:8000/v1').
            api_key: API authorization key. Defaults to env `OPENAI_API_KEY` or 'EMPTY'.
            base_url: Alias for `api_base`.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retry attempts on network or rate-limit errors.
            retry_delay: Base delay for exponential backoff (in seconds).
            max_workers: Maximum worker threads for parallel batch requests.
            default_headers: Custom HTTP headers to include in every request.
            use_legacy_completions: If True, uses `/completions` instead of `/chat/completions`
                for raw text `generate()` calls.
            extra_body: Arbitrary dictionary to merge into every request JSON body.
        """
        super().__init__(model_id=model_id, **kwargs)

        url = api_base or base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        self.api_base = self._normalize_base_url(url)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            # For local endpoints, vLLM and Ollama accept "EMPTY" or arbitrary tokens
            self.api_key = "EMPTY"

        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay
        self.max_workers = max(1, max_workers)
        self.default_headers = default_headers or {}
        self.use_legacy_completions = use_legacy_completions
        self.extra_body = extra_body or {}

        self._session = self._create_session()

    def _normalize_base_url(self, url: str) -> str:
        """Normalize URL ensuring proper protocol and no trailing slash."""
        url = url.strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"
        # If url has no path component or lacks /v1, add /v1 for OpenAI compatibility
        # e.g., "http://localhost:8000" -> "http://localhost:8000/v1"
        parts = url.split("://", 1)[-1].split("/")
        if len(parts) == 1:
            url = f"{url}/v1"
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

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SAAGA-OpenAIProvider/1.0",
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.api_key == "EMPTY":
            headers["Authorization"] = "Bearer EMPTY"
        headers.update(self.default_headers)
        return headers

    def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute POST request with exponential backoff retry on transient errors."""
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        headers = self._build_headers()

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

                # Retry on rate limit (429) or server errors (5xx)
                if response.status_code in (429, 500, 502, 503, 504):
                    err_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(
                        "Transient error from %s (attempt %d/%d): %s",
                        url,
                        attempt,
                        self.max_retries,
                        err_msg,
                    )
                    last_err = RuntimeError(err_msg)
                else:
                    # Client errors (400, 401, 403, 404, etc.) are unrecoverable
                    raise RuntimeError(
                        f"OpenAI API error at {url} (status {response.status_code}): {response.text}"
                    )
            except (requests.ConnectionError, requests.Timeout) as err:
                logger.warning(
                    "Network error connecting to %s (attempt %d/%d): %s",
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
            f"Failed after {self.max_retries} attempts to call {url}: {last_err}"
        ) from last_err

    def _extract_generation_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract and map common generation kwargs into OpenAI request params."""
        params: dict[str, Any] = {}
        if "temperature" in kwargs:
            params["temperature"] = float(kwargs["temperature"])
        if "top_p" in kwargs:
            params["top_p"] = float(kwargs["top_p"])
        if "max_tokens" in kwargs:
            params["max_tokens"] = int(kwargs["max_tokens"])
        elif "max_new_tokens" in kwargs:
            params["max_tokens"] = int(kwargs["max_new_tokens"])
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]
        if "presence_penalty" in kwargs:
            params["presence_penalty"] = float(kwargs["presence_penalty"])
        if "frequency_penalty" in kwargs:
            params["frequency_penalty"] = float(kwargs["frequency_penalty"])
        if "seed" in kwargs:
            params["seed"] = int(kwargs["seed"])

        # Forward any additional OpenAI-specific options (e.g. logprobs, response_format)
        for k in ("response_format", "logit_bias", "n", "user"):
            if k in kwargs:
                params[k] = kwargs[k]

        return params

    def chat(self, messages: list[dict[str, Any] | Message], **kwargs: Any) -> LLMResponse:
        """Send a chat request to `/chat/completions`."""
        norm_messages = normalize_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": norm_messages,
        }
        payload.update(self._extract_generation_params(kwargs))
        payload.update(self.extra_body)

        data = self._post_with_retry("chat/completions", payload)

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"Empty choices received from OpenAI endpoint: {data}")

        choice = choices[0]
        msg = choice.get("message", {})
        text = msg.get("content", "") or ""
        finish_reason = choice.get("finish_reason") or "stop"

        usage = data.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

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
        """Execute batch chat requests concurrently using a ThreadPoolExecutor."""
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
        """Send a raw completion prompt.

        If `use_legacy_completions` is True, sends to `/completions`.
        Otherwise, wraps prompt as a user message and sends to `/chat/completions`.
        """
        if not self.use_legacy_completions:
            return self.chat([{"role": "user", "content": prompt}], **kwargs)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
        }
        payload.update(self._extract_generation_params(kwargs))
        payload.update(self.extra_body)

        data = self._post_with_retry("completions", payload)

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"Empty choices received from OpenAI endpoint: {data}")

        choice = choices[0]
        text = choice.get("text", "") or ""
        finish_reason = choice.get("finish_reason") or "stop"

        usage = data.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            text=text,
            raw=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[LLMResponse]:
        """Execute batch raw completions concurrently."""
        if not prompts:
            return []
        if not self.use_legacy_completions:
            messages_batch = [[{"role": "user", "content": p}] for p in prompts]
            return self.chat_batch(messages_batch, **kwargs)

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
