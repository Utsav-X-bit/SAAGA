#!/usr/bin/env bash
# =============================================================================
# SAAGA Compute Node Evaluation: Llama-3-8B then Mistral-7B-Instruct-v0.2
# Runs 100% on compute node scn12 using local A100 GPUs
# =============================================================================
set -e

SAAGA_ROOT="/nlsasfs/home/isea/isea31/SAAGA"
cd "$SAAGA_ROOT"
export PYTHONPATH="$SAAGA_ROOT/src:$SAAGA_ROOT"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export SAAGA_EVALUATOR_DEVICE="cuda:2"

VLLM_BIN="/nlsasfs/home/isea/isea31/inference/vllm/qwen/.venv/bin/vllm"
PYTHON_BIN="/nlsasfs/home/isea/isea31/inference/vllm/qwen/.venv/bin/python3"

# Model Snapshot Paths
LLAMA3_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
MISTRAL_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
LEXI_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2/snapshots/f4617caeabd21f1820ac89bd125c80eda70901a7"

LOGS_DIR="$SAAGA_ROOT/logs"
mkdir -p "$LOGS_DIR" results/eval

echo "=============================================================================="
echo "SAAGA Compute Node Evaluation — Host: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "Start Time: $(date)"
echo "=============================================================================="

# Cleanup any existing vLLM instances on ports 8000 & 8001
echo "[*] Cleaning up any previous server instances..."
pkill -f "vllm serve.*8000" 2>/dev/null || true
pkill -f "vllm serve.*8001" 2>/dev/null || true
sleep 3

# -----------------------------------------------------------------------------
# 1. Start Attacker Model (Lexi-Uncensored-V2) on GPU 1 (Port 8001)
# -----------------------------------------------------------------------------
echo "[*] Launching Attacker Model (Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2) on GPU 1 (Port 8001)..."
CUDA_VISIBLE_DEVICES=1 nohup "$VLLM_BIN" serve "$LEXI_PATH" \
  --port 8001 --host 127.0.0.1 \
  --served-model-name Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager > "$LOGS_DIR/vllm_attacker_8001.log" 2>&1 &

ATTACKER_PID=$!
echo "[*] Attacker vLLM PID: $ATTACKER_PID. Waiting for readiness..."

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8001/v1/models | grep -q "Orenguteng"; then
    echo "[✓] Attacker Model is READY on port 8001!"
    break
  fi
  sleep 3
done

# -----------------------------------------------------------------------------
# 2. Evaluate Victim 1: Meta-Llama-3-8B-Instruct on GPU 0 (Port 8000)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "[EVAL 1/2] Meta-Llama-3-8B-Instruct"
echo "=============================================================================="

echo "[*] Launching Llama-3-8B-Instruct on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_BIN" serve "$LLAMA3_PATH" \
  --port 8000 --host 127.0.0.1 \
  --served-model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager > "$LOGS_DIR/vllm_llama3_8000.log" 2>&1 &

LLAMA_PID=$!
echo "[*] Llama-3 vLLM PID: $LLAMA_PID. Waiting for readiness..."

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8000/v1/models | grep -q "Meta-Llama"; then
    echo "[✓] Llama-3-8B is READY on port 8000!"
    break
  fi
  sleep 3
done

echo "[*] Starting SAAGA evaluation for Meta-Llama-3-8B-Instruct..."
"$PYTHON_BIN" -m saaga evaluate \
  -m meta-llama/Meta-Llama-3-8B-Instruct \
  -p openai -u http://127.0.0.1:8000/v1 \
  --base-model-url http://127.0.0.1:8001/v1 \
  --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --base-model-provider openai \
  -d data/TensorTrust_subsets \
  --samples-per-tier 100 \
  --max-attempts 20 \
  --mode adaptive \
  --max-parallel 8 \
  --provider-workers 8 \
  --provider-timeout 120 \
  --seed 42 \
  -o results/eval/scn12_llama3-8b | tee "$LOGS_DIR/eval_scn12_llama3.log"

echo "[✓] Finished Meta-Llama-3-8B-Instruct evaluation."

# Shutdown Victim 1
echo "[*] Shutting down Llama-3-8B vLLM..."
kill "$LLAMA_PID" 2>/dev/null || true
pkill -f "vllm serve.*8000" 2>/dev/null || true
sleep 5

# -----------------------------------------------------------------------------
# 3. Evaluate Victim 2: Mistral-7B-Instruct-v0.2 on GPU 0 (Port 8000)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "[EVAL 2/2] Mistral-7B-Instruct-v0.2"
echo "=============================================================================="

echo "[*] Launching Mistral-7B-Instruct-v0.2 on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_BIN" serve "$MISTRAL_PATH" \
  --port 8000 --host 127.0.0.1 \
  --served-model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager > "$LOGS_DIR/vllm_mistral_8000.log" 2>&1 &

MISTRAL_PID=$!
echo "[*] Mistral-7B vLLM PID: $MISTRAL_PID. Waiting for readiness..."

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8000/v1/models | grep -q "Mistral"; then
    echo "[✓] Mistral-7B-Instruct-v0.2 is READY on port 8000!"
    break
  fi
  sleep 3
done

echo "[*] Starting SAAGA evaluation for Mistral-7B-Instruct-v0.2..."
"$PYTHON_BIN" -m saaga evaluate \
  -m mistralai/Mistral-7B-Instruct-v0.2 \
  -p openai -u http://127.0.0.1:8000/v1 \
  --base-model-url http://127.0.0.1:8001/v1 \
  --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --base-model-provider openai \
  -d data/TensorTrust_subsets \
  --samples-per-tier 100 \
  --max-attempts 20 \
  --mode adaptive \
  --max-parallel 8 \
  --provider-workers 8 \
  --provider-timeout 120 \
  --seed 42 \
  -o results/eval/scn12_mistral7b | tee "$LOGS_DIR/eval_scn12_mistral7b.log"

echo "[✓] Finished Mistral-7B-Instruct-v0.2 evaluation."

# Teardown remaining servers
echo "[*] Cleaning up server processes..."
kill "$MISTRAL_PID" 2>/dev/null || true
kill "$ATTACKER_PID" 2>/dev/null || true
pkill -f "vllm serve.*8000" 2>/dev/null || true
pkill -f "vllm serve.*8001" 2>/dev/null || true

echo ""
echo "=============================================================================="
echo "ALL EVALUATIONS COMPLETE ON COMPUTE NODE scn12!"
echo "End Time: $(date)"
echo "=============================================================================="
