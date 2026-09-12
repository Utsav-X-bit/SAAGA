#!/usr/bin/env bash
set -e
cd /nlsasfs/home/isea/isea31/SAAGA
export PYTHONPATH="/nlsasfs/home/isea/isea31/SAAGA/src:/nlsasfs/home/isea/isea31/SAAGA"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export SAAGA_EVALUATOR_DEVICE="cuda:2"

PYTHON_BIN="/nlsasfs/home/isea/isea31/inference/vllm/qwen/.venv/bin/python3"

echo "[*] Starting SAAGA evaluation for Mistral-7B-Instruct-v0.2..."
"$PYTHON_BIN" -m saaga evaluate \
  -m "/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" \
  -p openai -u http://127.0.0.1:8000/v1 \
  --base-model-url http://127.0.0.1:8001/v1 \
  --base-model "/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2/snapshots/f4617caeabd21f1820ac89bd125c80eda70901a7" \
  --base-model-provider openai \
  -d data/TensorTrust_subsets \
  --samples-per-tier 100 \
  --max-attempts 20 \
  --mode adaptive \
  --max-parallel 8 \
  --provider-workers 8 \
  --provider-timeout 120 \
  --seed 42 \
  -o results/eval/scn12_mistral7b
