"""Provider registry and automatic backend resolution for SAAGA.

Allows retrieving LLM backends by provider type string or auto-detecting
the backend based on model paths, API URLs, and environment availability.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from saaga.providers.base import BaseLLMProvider, MockLLMProvider
from saaga.providers.hf_provider import HFProvider
from saaga.providers.ollama_provider import OllamaProvider
from saaga.providers.openai_provider import OpenAIProvider
from saaga.providers.vllm_provider import VLLMProvider

logger = logging.getLogger(__name__)

# Registry mapping canonical names and aliases to provider classes
_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "openai-compatible": OpenAIProvider,
    "vllm-server": OpenAIProvider,
    "openrouter": OpenAIProvider,
    "deepseek": OpenAIProvider,
    "groq": OpenAIProvider,
    "together": OpenAIProvider,
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
    "vllm-local": VLLMProvider,
    "vllm-engine": VLLMProvider,
    "hf": HFProvider,
    "huggingface": HFProvider,
    "transformers": HFProvider,
    "mock": MockLLMProvider,
    "dummy": MockLLMProvider,
    "test": MockLLMProvider,
}


def register_provider(name: str, provider_cls: type[BaseLLMProvider]) -> None:
    """Register a custom LLM provider class in the global registry.

    Args:
        name: Name or alias to identify the provider.
        provider_cls: Subclass of `BaseLLMProvider`.
    """
    if not issubclass(provider_cls, BaseLLMProvider):
        raise TypeError(f"Provider class must inherit from BaseLLMProvider, got {provider_cls}")
    _REGISTRY[name.strip().lower()] = provider_cls
    logger.info("Registered LLM provider '%s' -> %s", name, provider_cls.__name__)


def list_providers() -> list[str]:
    """Return a sorted list of registered provider names and aliases."""
    return sorted(_REGISTRY.keys())


def detect_provider_type(
    provider_type: str | None = None,
    model_id: str = "",
    api_base: str | None = None,
    **kwargs: Any,
) -> str:
    """Infer the appropriate provider backend from configuration parameters.

    Detection precedence:
    1. Explicit `provider_type` (if not 'auto' or None)
    2. API URL inspection (`api_base`)
    3. Model identifier patterns (`model_id`)
    4. Supplied engine objects in kwargs (`llm` -> vllm, `model` -> hf)
    5. Environment capability (vLLM if CUDA available, HF if torch available, else OpenAI)
    """
    if provider_type and provider_type.strip().lower() not in ("auto", "none", ""):
        return provider_type.strip().lower()

    # 1. Check if engine objects are passed directly
    if kwargs.get("llm") is not None:
        return "vllm"
    if kwargs.get("model") is not None or kwargs.get("tokenizer") is not None:
        return "hf"

    # 2. Check API URL
    if api_base:
        clean_url = api_base.strip().lower()
        if ":11434" in clean_url and "/v1" not in clean_url:
            return "ollama"
        if clean_url.startswith("http://") or clean_url.startswith("https://"):
            return "openai"

    # 3. Check model ID prefixes and keywords
    mid = model_id.strip().lower()
    if mid in ("mock", "test", "dummy") or mid.startswith("mock-") or mid.startswith("mock/"):
        return "mock"
    if mid.startswith("ollama/") or mid.startswith("ollama:"):
        return "ollama"
    if mid.startswith("openai/") or mid.startswith("gpt-"):
        return "openai"

    # 4. Environment-based auto-detection
    # Check if vLLM and CUDA are present
    try:
        import torch
        import vllm  # noqa: F401

        if torch.cuda.is_available():
            return "vllm"
    except Exception:
        pass

    # Check if PyTorch / Transformers is present
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return "hf"
    except Exception:
        pass

    # Fallback default: OpenAI-compatible endpoint
    return "openai"


def get_provider(
    provider_type: str | None = None,
    model_id: str = "",
    api_base: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Instantiate and return the appropriate LLM provider.

    Args:
        provider_type: Optional provider name (e.g. 'openai', 'ollama', 'vllm', 'hf', 'mock').
            If None or 'auto', automatically detected.
        model_id: Target model identifier or weights directory.
        api_base: Base URL for HTTP-based providers (OpenAI, Ollama).
        api_key: Optional API authentication key.
        **kwargs: Additional parameters forwarded to the provider constructor.

    Returns:
        Configured instance of `BaseLLMProvider`.
    """
    resolved_type = detect_provider_type(
        provider_type=provider_type,
        model_id=model_id,
        api_base=api_base,
        **kwargs,
    )

    if resolved_type not in _REGISTRY:
        raise ValueError(
            f"Unknown provider type '{resolved_type}'. "
            f"Available providers: {list_providers()}"
        )

    provider_cls = _REGISTRY[resolved_type]

    # Clean up model_id if prefixed with provider (e.g. 'ollama/llama3' -> 'llama3')
    clean_model_id = model_id
    if clean_model_id.startswith(f"{resolved_type}/"):
        clean_model_id = clean_model_id[len(resolved_type) + 1 :]
    elif clean_model_id.startswith(f"{resolved_type}:"):
        clean_model_id = clean_model_id[len(resolved_type) + 1 :]

    # Construct appropriate provider instance
    if provider_cls is OpenAIProvider:
        return OpenAIProvider(
            model_id=clean_model_id,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )
    elif provider_cls is OllamaProvider:
        return OllamaProvider(
            model_id=clean_model_id,
            api_base=api_base,
            **kwargs,
        )
    elif provider_cls is VLLMProvider:
        return VLLMProvider(
            model_id=clean_model_id,
            **kwargs,
        )
    elif provider_cls is HFProvider:
        return HFProvider(
            model_id=clean_model_id,
            **kwargs,
        )
    elif provider_cls is MockLLMProvider:
        return MockLLMProvider(
            model_id=clean_model_id,
            **kwargs,
        )
    else:
        # Custom user-registered provider class
        init_kwargs = dict(kwargs)
        if api_base is not None:
            init_kwargs["api_base"] = api_base
        if api_key is not None:
            init_kwargs["api_key"] = api_key
        return provider_cls(model_id=clean_model_id, **init_kwargs)
