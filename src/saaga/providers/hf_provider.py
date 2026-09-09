"""HuggingFace Transformers provider for SAAGA.

Wraps `AutoModelForCausalLM` and `AutoTokenizer` for local model inference.
Features:
- Lazy import of `torch` and `transformers` for environments without GPU/deep-learning libraries.
- Support for PEFT/LoRA adapter loading and dynamic switching.
- Device placement and precision handling (CUDA, CPU, MPS, bfloat16, float16, 8-bit/4-bit quantization).
- Left-padded batch generation for decoder-only architectures.
- Seamless fallback to plain text prompts when chat templates are missing.
"""
from __future__ import annotations

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


def _get_hf() -> tuple[Any, Any, Any]:
    """Lazy import helper for PyTorch and Transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return torch, AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        raise ImportError(
            "PyTorch and HuggingFace Transformers are required to use HFProvider, "
            "but could not be imported. Please install them with: pip install torch transformers"
        ) from err


def _get_peft() -> Any | None:
    """Optional import helper for PEFT LoRA adapter library."""
    try:
        import peft
        from peft import PeftModel

        return PeftModel
    except ImportError:
        return None


class HFProvider(BaseLLMProvider):
    """In-process provider running HuggingFace causal language models."""

    def __init__(
        self,
        model_id: str,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str | None = None,
        torch_dtype: str | Any | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        lora_adapters: dict[str, str] | None = None,
        lazy_init: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HuggingFace Transformers provider.

        Args:
            model_id: HuggingFace hub model ID or local directory path.
            model: Optional pre-loaded `AutoModelForCausalLM` instance.
            tokenizer: Optional pre-loaded `AutoTokenizer` instance.
            device: Target device ('cuda', 'cpu', 'auto', or 'mps').
            torch_dtype: Model weights dtype ('auto', 'float16', 'bfloat16', or torch.dtype).
            load_in_8bit: Enable 8-bit quantization via bitsandbytes.
            load_in_4bit: Enable 4-bit quantization via bitsandbytes.
            trust_remote_code: Allow execution of custom code from model repo.
            lora_adapters: Mapping of role names to adapter checkpoint directories.
            lazy_init: If True, defers model weight loading until first inference call.
            model_kwargs: Extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained`.
        """
        super().__init__(model_id=model_id, **kwargs)

        self.device_str = device
        self.torch_dtype_arg = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs or {}
        self.extra_kwargs = kwargs

        self._model = model
        self._tokenizer = tokenizer
        self._lora_registry: dict[str, str] = {}
        self._active_lora: str | None = None

        if lora_adapters:
            for name, path in lora_adapters.items():
                self.register_lora(name, path)

        if not lazy_init and self._model is None:
            self._init_model()

    def _init_model(self) -> None:
        """Load the tokenizer and causal LM weights."""
        if self._model is not None and self._tokenizer is not None:
            return

        torch, AutoModelForCausalLM, AutoTokenizer = _get_hf()

        # 1. Determine device and precision
        if self.device_str is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = self.device_str

        dtype = self.torch_dtype_arg
        if isinstance(dtype, str):
            if dtype == "auto":
                dtype = "auto"
            elif dtype in ("float16", "fp16"):
                dtype = torch.float16
            elif dtype in ("bfloat16", "bf16"):
                dtype = torch.bfloat16
            elif dtype in ("float32", "fp32"):
                dtype = torch.float32

        # 2. Load tokenizer
        if self._tokenizer is None:
            logger.info("Loading HuggingFace tokenizer for %s", self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            # Ensure padding configuration is correct for decoder-only generation
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"

        # 3. Load model weights
        if self._model is None:
            logger.info("Loading HuggingFace model %s on %s", self.model_id, target_device)
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
            }
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if self.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                load_kwargs["load_in_4bit"] = True

            if target_device == "auto":
                load_kwargs["device_map"] = "auto"

            load_kwargs.update(self.model_kwargs)
            load_kwargs.update(self.extra_kwargs)

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

            if target_device != "auto" and not (self.load_in_8bit or self.load_in_4bit):
                model = model.to(target_device)

            model.eval()
            self._model = model

    @property
    def model(self) -> Any:
        """Access underlying model, initializing if needed."""
        if self._model is None:
            self._init_model()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Access underlying tokenizer, initializing if needed."""
        if self._tokenizer is None:
            self._init_model()
        return self._tokenizer

    def register_lora(self, name: str, path: str | Path) -> None:
        """Register a LoRA adapter checkpoint directory.

        Args:
            name: Identifier for the adapter role (e.g. 'planner', 'generator').
            path: Local checkpoint directory.
        """
        str_path = str(Path(path).resolve())
        self._lora_registry[name.lower()] = str_path
        logger.info("Registered HF LoRA adapter '%s' at %s", name, str_path)

        # If model is already loaded and PEFT is available, attach adapter now
        if self._model is not None:
            self._attach_lora(name.lower(), str_path)

    def _attach_lora(self, name: str, path: str) -> None:
        """Attach a LoRA adapter to the model using PEFT."""
        PeftModel = _get_peft()
        if PeftModel is None:
            logger.warning("PEFT library is not installed; cannot attach LoRA adapter '%s'", name)
            return

        try:
            if not isinstance(self._model, PeftModel):
                # Wrap base model as PeftModel with initial adapter
                logger.info("Wrapping base model with PeftModel (adapter: %s)", name)
                self._model = PeftModel.from_pretrained(self._model, path, adapter_name=name)
            else:
                logger.info("Adding adapter '%s' to existing PeftModel", name)
                self._model.load_adapter(path, adapter_name=name)
            self._active_lora = name
        except Exception as err:
            logger.error("Failed to load PEFT adapter '%s' from %s: %s", name, path, err)

    def _set_active_lora(self, name_or_path: str | None) -> None:
        """Switch active LoRA adapter on the PEFT model."""
        if not name_or_path:
            return

        name = name_or_path.lower()
        if name == self._active_lora:
            return

        PeftModel = _get_peft()
        if PeftModel is None or not isinstance(self._model, PeftModel):
            if name in self._lora_registry:
                self._attach_lora(name, self._lora_registry[name])
            elif Path(name_or_path).exists():
                self._attach_lora(name, str(Path(name_or_path).resolve()))
            return

        try:
            if hasattr(self._model, "set_adapter"):
                self._model.set_adapter(name)
                self._active_lora = name
                logger.debug("Switched active PEFT adapter to: %s", name)
        except Exception as err:
            logger.warning("Could not switch PEFT adapter to '%s': %s", name, err)

    def _format_messages(self, messages: list[dict[str, Any] | Message]) -> str:
        """Format messages using the tokenizer's chat template or text fallback."""
        norm_messages = normalize_messages(messages)
        tkr = self.tokenizer
        if hasattr(tkr, "apply_chat_template"):
            try:
                return tkr.apply_chat_template(
                    norm_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as err:
                logger.debug("Chat template application failed: %s; using plain text fallback", err)

        return messages_to_prompt(norm_messages)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate completion for a single text prompt."""
        results = self.generate_batch([prompt], **kwargs)
        return results[0]

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[LLMResponse]:
        """Generate completions for a batch of text prompts using HuggingFace generate."""
        if not prompts:
            return []

        torch, _, _ = _get_hf()

        # Handle LoRA adapter selection
        lora_name = kwargs.pop("lora_name", None) or kwargs.pop("lora", None)
        if lora_name:
            self._set_active_lora(lora_name)

        # Parse generation hyper-parameters
        max_new_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens", 256))
        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs["top_p"]) if "top_p" in kwargs else None
        top_k = int(kwargs["top_k"]) if "top_k" in kwargs else None
        repetition_penalty = (
            float(kwargs["repetition_penalty"]) if "repetition_penalty" in kwargs else None
        )

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False

        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        # Tokenize batch with left padding for decoder-only generation
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to model device
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        input_seq_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        responses: list[LLMResponse] = []
        for i in range(len(prompts)):
            # With left padding, generated new tokens start at input_seq_len
            generated_ids = outputs[i, input_seq_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            prompt_tokens = int(inputs["attention_mask"][i].sum().item())
            completion_tokens = len(generated_ids)

            output_ids_list = (
                generated_ids.tolist()
                if hasattr(generated_ids, "tolist")
                else list(generated_ids)
            )
            responses.append(
                LLMResponse(
                    text=text,
                    raw={"output_ids": output_ids_list},
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finish_reason="stop",
                )
            )

        return responses

    def chat(self, messages: list[dict[str, Any] | Message], **kwargs: Any) -> LLMResponse:
        """Execute single multi-turn chat generation."""
        prompt = self._format_messages(messages)
        return self.generate(prompt, **kwargs)

    def chat_batch(
        self, messages_batch: list[list[dict[str, Any] | Message]], **kwargs: Any
    ) -> list[LLMResponse]:
        """Execute batched multi-turn chat generations."""
        if not messages_batch:
            return []
        prompts = [self._format_messages(msgs) for msgs in messages_batch]
        return self.generate_batch(prompts, **kwargs)
