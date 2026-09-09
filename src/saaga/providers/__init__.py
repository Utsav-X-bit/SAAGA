"""SAAGA LLM Provider Interfaces and Implementations.

Universal provider abstraction supporting:
- OpenAI-compatible HTTP servers (vLLM server, Ollama v1, cloud APIs)
- Native Ollama REST API
- In-process vLLM engine with LoRA adapter management
- HuggingFace Transformers with PEFT LoRA switching
- Mock provider for offline testing and verification
"""
from __future__ import annotations

from saaga.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    MockLLMProvider,
    messages_to_prompt,
    normalize_messages,
)
from saaga.providers.hf_provider import HFProvider
from saaga.providers.ollama_provider import OllamaProvider
from saaga.providers.openai_provider import OpenAIProvider
from saaga.providers.registry import (
    detect_provider_type,
    get_provider,
    list_providers,
    register_provider,
)
from saaga.providers.vllm_provider import VLLMProvider

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "Message",
    "MockLLMProvider",
    "normalize_messages",
    "messages_to_prompt",
    "OpenAIProvider",
    "OllamaProvider",
    "VLLMProvider",
    "HFProvider",
    "get_provider",
    "register_provider",
    "list_providers",
    "detect_provider_type",
]
