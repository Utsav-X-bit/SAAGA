"""In-process vLLM engine provider for SAAGA.

Provides direct access to vLLM's high-throughput offline inference engine (`LLM`, `SamplingParams`).
Features:
- Lazy import of `vllm` and `torch` so non-GPU environments can still run other providers.
- Seamless LoRA adapter management for shared base models (e.g. planner and generator adapters).
- Engine sharing: accepts a pre-instantiated `vllm.LLM` instance or creates a new one.
- Native parallel batch generation across GPU workers.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Sequence

from saaga.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    messages_to_prompt,
    normalize_messages,
)

logger = logging.getLogger(__name__)


def _get_vllm() -> tuple[Any, Any, Any, Any]:
    """Lazy import helper for vLLM modules.

    Returns:
        tuple of (vllm_module, LLM_class, SamplingParams_class, LoRARequest_class)
    Raises:
        ImportError: If vLLM is not installed.
    """
    try:
        import vllm
        from vllm import LLM, SamplingParams

        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            LoRARequest = None  # Older vLLM without LoRA support

        return vllm, LLM, SamplingParams, LoRARequest
    except ImportError as err:
        raise ImportError(
            "vLLM is required to use VLLMProvider, but it could not be imported. "
            "Please ensure vLLM is installed (`pip install vllm`) and CUDA is available."
        ) from err


class VLLMProvider(BaseLLMProvider):
    """Provider executing inference in-process via vLLM."""

    def __init__(
        self,
        model_id: str,
        llm: Any | None = None,
        tokenizer: Any | None = None,
        enable_lora: bool = False,
        max_lora_rank: int = 128,
        max_loras: int = 4,
        max_cpu_loras: int = 8,
        lora_extra_vocab_size: int = 256,
        lora_adapters: dict[str, str] | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        kv_cache_dtype: str = "auto",
        max_num_seqs: int = 256,
        enforce_eager: bool = False,
        quantization: str | None = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        lazy_init: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the in-process vLLM engine provider.

        Args:
            model_id: Path to local model weights or HuggingFace repo ID.
            llm: Optional pre-instantiated `vllm.LLM` instance. If provided,
                engine creation is skipped.
            tokenizer: Optional pre-loaded tokenizer.
            enable_lora: Whether to enable LoRA adapters on the engine.
            max_lora_rank: Maximum LoRA rank supported by the engine.
            max_loras: Maximum number of active LoRA adapters in GPU memory.
            max_cpu_loras: Maximum number of cached LoRAs in CPU memory.
            lora_extra_vocab_size: Extra vocabulary size for LoRA adapters.
            lora_adapters: Mapping of role names to adapter checkpoint directories.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU VRAM allocated to this engine.
            max_model_len: Context window override (tokens).
            kv_cache_dtype: KV cache precision ('auto', 'fp8_e5m2', etc.).
            max_num_seqs: Maximum sequences per batch iteration.
            enforce_eager: Disables CUDA graph capture when True (saves memory).
            quantization: Quantization method (e.g. 'bitsandbytes', 'awq').
            trust_remote_code: Allow execution of custom code from model repo.
            dtype: Model weight precision ('auto', 'float16', 'bfloat16').
            lazy_init: If True, delays engine creation until the first inference call.
            **kwargs: Extra parameters forwarded to `vllm.LLM(...)`.
        """
        super().__init__(model_id=model_id, **kwargs)

        self.enable_lora = enable_lora or bool(lora_adapters)
        self.max_lora_rank = max_lora_rank
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.lora_extra_vocab_size = lora_extra_vocab_size
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.kv_cache_dtype = kv_cache_dtype
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.extra_vllm_kwargs = kwargs

        self._llm = llm
        self._tokenizer = tokenizer
        self._lora_registry: dict[str, Any] = {}
        self._next_lora_id = 1

        # Register any supplied adapters
        if lora_adapters:
            for name, path in lora_adapters.items():
                self.register_lora(name, path)

        if not lazy_init and self._llm is None:
            self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the vLLM engine if not already created."""
        if self._llm is not None:
            return

        _, LLM, _, _ = _get_vllm()

        llm_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
        }

        if self.enable_lora:
            llm_kwargs.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": self.max_lora_rank,
                    "max_loras": self.max_loras,
                    "max_cpu_loras": self.max_cpu_loras,
                    "lora_extra_vocab_size": self.lora_extra_vocab_size,
                }
            )

        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.kv_cache_dtype != "auto":
            llm_kwargs["kv_cache_dtype"] = self.kv_cache_dtype
        if self.max_num_seqs != 256:
            llm_kwargs["max_num_seqs"] = self.max_num_seqs
        if self.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if self.quantization:
            llm_kwargs["quantization"] = self.quantization

        llm_kwargs.update(self.extra_vllm_kwargs)

        logger.info("Initializing vLLM engine for model %s with kwargs: %s", self.model_id, llm_kwargs)
        self._llm = LLM(**llm_kwargs)

    @property
    def llm(self) -> Any:
        """Access the underlying vLLM LLM instance, initializing if needed."""
        if self._llm is None:
            self._init_engine()
        return self._llm

    @property
    def tokenizer(self) -> Any:
        """Access the tokenizer associated with this engine."""
        if self._tokenizer is None:
            self._tokenizer = self.llm.get_tokenizer()
        return self._tokenizer

    def register_lora(self, name: str, path: str | Path, lora_id: int | None = None) -> Any:
        """Register a LoRA adapter checkpoint with the engine.

        Args:
            name: Friendly identifier for the adapter (e.g. 'planner', 'generator').
            path: Local path to the adapter checkpoint directory.
            lora_id: Unique integer ID for the adapter. Auto-assigned if None.

        Returns:
            The created `LoRARequest` object.
        """
        _, _, _, LoRARequest = _get_vllm()
        if LoRARequest is None:
            raise RuntimeError("vLLM does not support LoRARequest in this environment.")

        str_path = str(Path(path).resolve())
        int_id = lora_id if lora_id is not None else self._next_lora_id
        self._next_lora_id = max(self._next_lora_id, int_id + 1)

        req_kwargs: dict[str, Any] = {
            "lora_name": name,
            "lora_int_id": int_id,
            "lora_path": str_path,
        }

        # vLLM >= 0.8.5 supports base_model_name
        try:
            if "base_model_name" in inspect.signature(LoRARequest.__init__).parameters:
                req_kwargs["base_model_name"] = self.model_id
            request = LoRARequest(**req_kwargs)
        except TypeError:
            request = LoRARequest(name, int_id, str_path)

        self._lora_registry[name.lower()] = request
        logger.info("Registered LoRA adapter '%s' (id=%d) from %s", name, int_id, str_path)
        return request

    def get_lora_request(self, name_or_path: str | None) -> Any:
        """Retrieve a registered `LoRARequest` or create one dynamically."""
        if not name_or_path:
            return None

        # Check by registry name
        key = name_or_path.lower()
        if key in self._lora_registry:
            return self._lora_registry[key]

        # Check if the string points to an existing directory
        p = Path(name_or_path)
        if p.exists() and p.is_dir():
            return self.register_lora(p.name, p)

        raise KeyError(
            f"LoRA adapter '{name_or_path}' is neither registered nor a valid directory. "
            f"Available adapters: {list(self._lora_registry.keys())}"
        )

    def _resolve_lora(self, kwargs: dict[str, Any]) -> Any:
        """Resolve LoRA request from kwargs ('lora_request', 'lora_name', or 'lora')."""
        if "lora_request" in kwargs and kwargs["lora_request"] is not None:
            return kwargs.pop("lora_request")

        lora_name = kwargs.pop("lora_name", None) or kwargs.pop("lora", None)
        if lora_name is not None:
            return self.get_lora_request(lora_name)

        return None

    def _build_sampling_params(self, kwargs: dict[str, Any]) -> Any:
        """Construct a `vllm.SamplingParams` object from generation kwargs."""
        _, _, SamplingParams, _ = _get_vllm()

        # If already a SamplingParams object, return directly
        if "sampling_params" in kwargs and isinstance(kwargs["sampling_params"], SamplingParams):
            return kwargs.pop("sampling_params")

        params: dict[str, Any] = {}

        if "temperature" in kwargs:
            params["temperature"] = float(kwargs["temperature"])
        else:
            params["temperature"] = 0.7

        if "top_p" in kwargs:
            params["top_p"] = float(kwargs["top_p"])
        if "top_k" in kwargs:
            params["top_k"] = int(kwargs["top_k"])
        if "max_tokens" in kwargs:
            params["max_tokens"] = int(kwargs["max_tokens"])
        elif "max_new_tokens" in kwargs:
            params["max_tokens"] = int(kwargs["max_new_tokens"])
        else:
            params["max_tokens"] = 256

        if "stop" in kwargs:
            stop = kwargs["stop"]
            params["stop"] = [stop] if isinstance(stop, str) else list(stop)

        if "presence_penalty" in kwargs:
            params["presence_penalty"] = float(kwargs["presence_penalty"])
        if "frequency_penalty" in kwargs:
            params["frequency_penalty"] = float(kwargs["frequency_penalty"])
        if "repetition_penalty" in kwargs:
            params["repetition_penalty"] = float(kwargs["repetition_penalty"])
        if "seed" in kwargs:
            params["seed"] = int(kwargs["seed"])

        return SamplingParams(**params)

    def _format_messages(self, messages: list[dict[str, Any] | Message]) -> str:
        """Format a list of messages into a single prompt string."""
        norm_messages = normalize_messages(messages)
        tkr = self.tokenizer
        if hasattr(tkr, "apply_chat_template"):
            try:
                return tkr.apply_chat_template(
                    norm_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as err:
                logger.debug("Chat template application failed: %s; falling back to plain text", err)

        return messages_to_prompt(norm_messages)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate completion for a single text prompt."""
        results = self.generate_batch([prompt], **kwargs)
        return results[0]

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[LLMResponse]:
        """Generate completions for a batch of text prompts using native vLLM batching."""
        if not prompts:
            return []

        lora_req = self._resolve_lora(kwargs)
        sampling_params = self._build_sampling_params(kwargs)

        gen_kwargs: dict[str, Any] = {
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
        if lora_req is not None:
            gen_kwargs["lora_request"] = lora_req

        outputs = self.llm.generate(prompts, **gen_kwargs)

        responses: list[LLMResponse] = []
        for out in outputs:
            generated_text = out.outputs[0].text if out.outputs else ""
            finish_reason = out.outputs[0].finish_reason if out.outputs else "stop"
            prompt_tokens = len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else 0
            completion_tokens = len(out.outputs[0].token_ids) if out.outputs else 0

            responses.append(
                LLMResponse(
                    text=generated_text,
                    raw={"request_output": out},
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finish_reason=finish_reason or "stop",
                )
            )

        return responses

    def chat(self, messages: list[dict[str, Any] | Message], **kwargs: Any) -> LLMResponse:
        """Process a single chat interaction."""
        prompt = self._format_messages(messages)
        return self.generate(prompt, **kwargs)

    def chat_batch(
        self, messages_batch: list[list[dict[str, Any] | Message]], **kwargs: Any
    ) -> list[LLMResponse]:
        """Process a batch of chat interactions concurrently via vLLM batching."""
        if not messages_batch:
            return []
        prompts = [self._format_messages(msgs) for msgs in messages_batch]
        return self.generate_batch(prompts, **kwargs)
