"""
SAAGA Model Setup & Cloud Downloader
====================================
Utility for downloading and setting up trained models, LoRA adapters,
reward models, and ancillary tokenizers from HuggingFace Hub or cloud storage.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_MODELS = {
    "victim": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "type": "causal_lm",
        "description": "Default target victim model",
    },
    "base_lora": {
        "repo_id": "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        "type": "causal_lm",
        "description": "Shared uncensored base model for planner and generator LoRA adapters",
    },
    "judge": {
        "repo_id": "distilbert/distilbert-base-uncased",
        "type": "classifier",
        "description": "DistilBERT stop-point identifier classifier",
    },
    "embedding": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "embedding",
        "description": "Sentence embedding model for RAG defense retrieval",
    },
    "translation": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "type": "seq2seq",
        "description": "NLLB-200 distilled offline translation model for mutation fallback",
    },
}

COMPONENT_ALIASES: dict[str, str] = {
    "tl-mutator": "translation",
    "tl_mutator": "translation",
    "tl": "translation",
    "translation": "translation",
    "nllb": "translation",
    "base_lora": "base_lora",
    "base-lora": "base_lora",
    "base-model": "base_lora",
    "base_model": "base_lora",
    "lexi": "base_lora",
    "victim": "victim",
    "llama3": "victim",
    "llama-3": "victim",
    "judge": "judge",
    "pi_reward_model": "judge",
    "embedding": "embedding",
    "rag": "embedding",
}


def resolve_component_name(name: str) -> str:
    """Resolve component name or alias to canonical DEFAULT_MODELS key."""
    cleaned = str(name).strip().lower().replace("_", "-")
    return COMPONENT_ALIASES.get(cleaned, COMPONENT_ALIASES.get(name.strip().lower(), name.strip().lower()))
def download_model(
    repo_id: str,
    target_dir: str | Path | None = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> Path:
    """Download a model from Hugging Face Hub to a local directory or HF cache.

    Args:
        repo_id: HuggingFace repository ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        target_dir: Optional custom local directory to store weights
        hf_token: Optional Hugging Face access token
        allow_patterns: Optional list of file glob patterns to filter downloads

    Returns:
        Path to the downloaded model directory
    """
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install it via: pip install huggingface_hub"
        )

    print(f"[*] Downloading {repo_id}...")
    local_dir = str(target_dir) if target_dir else None

    kwargs = {
        "repo_id": repo_id,
        "token": token,
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    path = snapshot_download(**kwargs)
    print(f"[✓] {repo_id} downloaded successfully to: {path}")
    return Path(path)


def setup_all_models(
    target_root: str | Path = "models",
    components: Optional[list[str]] = None,
    hf_token: Optional[str] = None,
) -> dict[str, Path]:
    """Download all required models for full offline SAAGA operation.

    Args:
        target_root: Directory where models should be installed
        components: List of components to download (e.g. ['judge', 'embedding']).
                    If None, downloads all default components except heavy victim models.
        hf_token: HuggingFace access token

    Returns:
        Dictionary mapping component names to local paths
    """
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    raw_selected = components or ["embedding", "translation"]
    selected = [resolve_component_name(c) for c in raw_selected]

    results = {}
    for name in selected:
        if name not in DEFAULT_MODELS:
            print(f"[!] Warning: Unknown model component '{name}'. Skipping.")
            print(f"    Available components: {list(DEFAULT_MODELS.keys()) + list(COMPONENT_ALIASES.keys())}")
            continue
        info = DEFAULT_MODELS[name]
        dest = root / name
        try:
            downloaded = download_model(info["repo_id"], target_dir=dest, hf_token=hf_token)
            results[name] = downloaded
        except Exception as exc:
            print(f"[X] Failed to download {name} ({info['repo_id']}): {exc}")

    return results
