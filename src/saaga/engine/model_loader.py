"""
SAAGA Engine Model Loader
=========================
Manages in-process GPU allocation, vLLM engine instantiation, LoRA adapter routing,
and auxiliary sequence classifier loading with resilient fallbacks.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_vllm_engine(
    model_id: str,
    gpu_memory_utilization: float = 0.50,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    enable_lora: bool = False,
    max_loras: int = 4,
    enforce_eager: bool = False,
    kv_cache_dtype: str = "auto",
    max_num_seqs: int = 32,
    dtype: str = "auto",
) -> Any:
    """Instantiate a local in-process vLLM engine on CUDA."""
    try:
        from vllm import LLM
    except ImportError as err:
        raise ImportError("vLLM is required to load an in-process engine. Run: pip install vllm") from err

    kwargs: dict[str, Any] = {
        "model": model_id,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": enforce_eager,
        "kv_cache_dtype": kv_cache_dtype,
        "max_num_seqs": max_num_seqs,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_loras"] = max_loras
        kwargs["max_lora_rank"] = 64

    logger.info("[ENGINE] Initializing vLLM engine for %s (GPU util: %.2f)...", model_id, gpu_memory_utilization)
    return LLM(**kwargs)


def load_classifier_model(
    checkpoint_path: str | Path,
    fallback_repo_id: str = "distilbert/distilbert-base-uncased",
    num_labels: int = 2,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Load a Hugging Face sequence classification model and tokenizer with graceful fallback.

    Returns (model, tokenizer). If checkpoint is absent, returns (None, None)
    so downstream components automatically engage their dummy heuristic fallback.
    """
    ckpt = Path(checkpoint_path)
    if not (ckpt.exists() and any(ckpt.iterdir())):
        logger.info("[LOADER] Checkpoint '%s' not found locally. Using placeholder heuristic.", checkpoint_path)
        return None, None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt),
            local_files_only=True,
            num_labels=num_labels,
        )
        if device != "cpu" and torch.cuda.is_available():
            model = model.to(device)
        model.eval()
        return model, tok
    except Exception as exc:
        logger.warning("[LOADER] Failed to load classifier from '%s' (%s). Using placeholder heuristic.", checkpoint_path, exc)
        return None, None
