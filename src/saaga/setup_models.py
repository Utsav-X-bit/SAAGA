"""
SAAGA Model & Data Cloud Downloader
===================================
Utility for downloading and setting up trained models, LoRA adapters,
access code predictors, reward models, and datasets from HuggingFace Hub
and Google Drive.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Default Google Drive public root folder for trained models and datasets
DEFAULT_GDRIVE_FOLDER_ID = os.environ.get(
    "SAAGA_GDRIVE_FOLDER_ID", "1BU6x9tzA9EPhMAIjaKlTAig85IZ3pSZY"
)

# HuggingFace Hub models
DEFAULT_MODELS = {
    "victim": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "type": "causal_lm",
        "description": "Default target victim model",
        "source": "hf",
    },
    "base_lora": {
        "repo_id": "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        "type": "causal_lm",
        "description": "Shared uncensored base model for planner and generator LoRA adapters",
        "source": "hf",
    },
    "judge": {
        "repo_id": "distilbert/distilbert-base-uncased",
        "type": "classifier",
        "description": "DistilBERT stop-point identifier classifier",
        "source": "hf",
    },
    "embedding": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "embedding",
        "description": "Sentence embedding model for RAG defense retrieval",
        "source": "hf",
    },
    "translation": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "type": "seq2seq",
        "description": "NLLB-200 distilled offline translation model for mutation fallback",
        "source": "hf",
    },
}

# Google Drive trained assets and datasets
GDRIVE_COMPONENTS = {
    "access-code-predictor": {
        "remote_path": "experiment/access_code_predictor",
        "default_dest": "experiment/access_code_predictor",
        "description": "Trained DistilBERT Access Code Predictor model and tokenizers",
        "source": "gdrive",
    },
    "ranker": {
        "remote_path": "models/ranker_deberta_v1",
        "default_dest": "models/ranker_deberta_v1",
        "description": "Trained DeBERTa-v3 Ranker model checkpoint",
        "source": "gdrive",
    },
    "defense-classifier": {
        "remote_path": "models/defense_classifier",
        "default_dest": "models/defense_classifier",
        "description": "Trained DistilBERT Defense Classifier checkpoints",
        "source": "gdrive",
    },
    "pi-reward-model": {
        "remote_path": "pre_trained/pi_reward_model",
        "default_dest": "pre_trained/pi_reward_model",
        "description": "Pre-trained Pi Reward Model / DistilBERT Stop Judge",
        "source": "gdrive",
    },
    "generator-lora": {
        "remote_path": "experiment/results/generator_sft_v2",
        "default_dest": "experiment/results/generator_sft_v2",
        "description": "Trained Generator SFT v2 LoRA adapter",
        "source": "gdrive",
    },
    "planner-lora": {
        "remote_path": "experiment/results/planner_sft_v2_contract_anchor",
        "default_dest": "experiment/results/planner_sft_v2_contract_anchor",
        "description": "Trained Planner SFT v2 Contract Anchor LoRA adapter",
        "source": "gdrive",
    },
    "planner-repair-lora": {
        "remote_path": "experiment/results/planner_sft_v2_contract_repair",
        "default_dest": "experiment/results/planner_sft_v2_contract_repair",
        "description": "Trained Planner SFT v2 Contract Repair LoRA adapter",
        "source": "gdrive",
    },
    "qlo-lora": {
        "remote_path": "experiment/results/qlo_curriculum_v1",
        "default_dest": "experiment/results/qlo_curriculum_v1",
        "description": "Trained QLO Curriculum v1 LoRA adapter",
        "source": "gdrive",
    },
    "data": {
        "remote_path": "data",
        "default_dest": "data",
        "description": "Large dataset files (.jsonl, .db, and TensorTrust subsets)",
        "source": "gdrive",
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
    "pi_reward_model": "pi-reward-model",
    "pi-reward": "pi-reward-model",
    "embedding": "embedding",
    "rag": "embedding",
    "access_code_predictor": "access-code-predictor",
    "access-code-predictor": "access-code-predictor",
    "ac-predictor": "access-code-predictor",
    "ac_predictor": "access-code-predictor",
    "predictor": "access-code-predictor",
    "ranker": "ranker",
    "ranker-deberta": "ranker",
    "defense_classifier": "defense-classifier",
    "defense-classifier": "defense-classifier",
    "generator_sft_v2": "generator-lora",
    "generator-lora": "generator-lora",
    "generator": "generator-lora",
    "planner_sft_v2": "planner-lora",
    "planner-lora": "planner-lora",
    "planner": "planner-lora",
    "planner-repair": "planner-repair-lora",
    "qlo": "qlo-lora",
    "qlo-lora": "qlo-lora",
    "dataset": "data",
    "datasets": "data",
    "data": "data",
}


def resolve_component_name(name: str) -> str:
    """Resolve component name or alias to canonical key."""
    cleaned = str(name).strip().lower().replace("_", "-")
    return COMPONENT_ALIASES.get(cleaned, COMPONENT_ALIASES.get(name.strip().lower(), name.strip().lower()))


def download_from_gdrive(
    remote_path: str,
    target_dir: str | Path,
    folder_id: str = DEFAULT_GDRIVE_FOLDER_ID,
) -> Path:
    """Download files from public Google Drive folder using rclone, gdown, or direct HTTP.

    Args:
        remote_path: Relative path inside the Google Drive root folder (e.g. 'experiment/access_code_predictor')
        target_dir: Local destination directory
        folder_id: Google Drive folder ID

    Returns:
        Path to local destination directory
    """
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading '{remote_path}' from Google Drive (Folder ID: {folder_id}) into '{dest}'...")

    # Method 1: rclone (if configured with gdrive remote or available)
    rclone_bin = shutil.which("rclone")
    if rclone_bin:
        cmd = [
            rclone_bin,
            "copy",
            f"gdrive:{remote_path}",
            str(dest),
            "--drive-root-folder-id",
            folder_id,
            "--stats",
            "10s",
            "--stats-one-line",
            "-v",
        ]
        try:
            print(f"[*] Running: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=False, check=False)
            if res.returncode == 0:
                print(f"[✓] Successfully downloaded '{remote_path}' to '{dest}' via rclone.")
                return dest
            else:
                print(f"[!] rclone exited with code {res.returncode}. Trying fallback...")
        except Exception as e:
            print(f"[!] rclone execution failed: {e}. Trying fallback...")

    # Method 2: gdown (Python tool for public Google Drive folders)
    gdown_bin = shutil.which("gdown") or str(Path.home() / ".local" / "bin" / "gdown")
    if Path(gdown_bin).exists() or shutil.which("gdown"):
        bin_path = gdown_bin if Path(gdown_bin).exists() else shutil.which("gdown")
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        cmd = [bin_path, "--folder", folder_url, "-O", str(dest), "--remaining-ok"]
        try:
            print(f"[*] Running: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=False, check=False)
            if res.returncode == 0:
                print(f"[✓] Successfully downloaded from Google Drive via gdown.")
                return dest
            else:
                print(f"[!] gdown exited with code {res.returncode}.")
        except Exception as e:
            print(f"[!] gdown execution failed: {e}.")

    # Method 3: Instruction banner if tools are missing
    print(f"\n[X] Could not automatically download Google Drive folder '{remote_path}'.")
    print(f"    Folder URL: https://drive.google.com/drive/folders/{folder_id}")
    print(f"    Please install rclone or gdown to enable automated Google Drive sync:")
    print(f"      pip install gdown")
    print(f"      gdown --folder https://drive.google.com/drive/folders/{folder_id} -O {dest}")
    return dest


def download_model(
    repo_id: str,
    target_dir: str | Path | None = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> Path:
    """Download a model from Hugging Face Hub to a local directory or HF cache."""
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading HuggingFace models. "
            "Install it via: pip install huggingface_hub"
        )

    print(f"[*] Downloading {repo_id} from HuggingFace Hub...")
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
    gdrive_folder_id: str = DEFAULT_GDRIVE_FOLDER_ID,
) -> dict[str, Path]:
    """Download all required models and components for full offline SAAGA operation.

    Args:
        target_root: Directory where models should be installed
        components: List of components to download
        hf_token: HuggingFace access token
        gdrive_folder_id: Google Drive public folder ID

    Returns:
        Dictionary mapping component names to local paths
    """
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    raw_selected = components or ["embedding", "translation", "access-code-predictor"]
    selected = [resolve_component_name(c) for c in raw_selected]

    results = {}
    for name in selected:
        # Check Google Drive components first
        if name in GDRIVE_COMPONENTS:
            info = GDRIVE_COMPONENTS[name]
            dest = Path(info["default_dest"])
            try:
                downloaded = download_from_gdrive(
                    remote_path=info["remote_path"],
                    target_dir=dest,
                    folder_id=gdrive_folder_id,
                )
                results[name] = downloaded
            except Exception as exc:
                print(f"[X] Failed to download {name} from Google Drive: {exc}")
            continue

        # Check HuggingFace components
        if name in DEFAULT_MODELS:
            info = DEFAULT_MODELS[name]
            dest = root / name
            try:
                downloaded = download_model(info["repo_id"], target_dir=dest, hf_token=hf_token)
                results[name] = downloaded
            except Exception as exc:
                print(f"[X] Failed to download {name} from HuggingFace ({info['repo_id']}): {exc}")
            continue

        print(f"[!] Warning: Unknown model component '{name}'. Skipping.")
        all_comps = list(DEFAULT_MODELS.keys()) + list(GDRIVE_COMPONENTS.keys())
        print(f"    Available components: {all_comps}")

    return results


def download_dataset(
    target_dir: str | Path = "data",
    folder_id: str = DEFAULT_GDRIVE_FOLDER_ID,
) -> Path:
    """Download the complete benchmark datasets from Google Drive."""
    return download_from_gdrive(remote_path="data", target_dir=target_dir, folder_id=folder_id)
