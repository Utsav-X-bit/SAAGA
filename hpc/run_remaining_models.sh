#!/usr/bin/env bash
# =============================================================================
# SAAGA Remaining Evaluations: InternLM-7B then Gemma-2B-IT
# Runs on scn12 (A100 GPUs) using local Lexi attacker on GPU 1 (Port 8001)
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

INTERNLM_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--internlm--internlm-chat-7b/snapshots/4e2d2e185058a4cbaa2377ba1bb993eb599ce492"
GEMMA_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
LEXI_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2/snapshots/f4617caeabd21f1820ac89bd125c80eda70901a7"

LOGS_DIR="$SAAGA_ROOT/logs"
mkdir -p "$LOGS_DIR" results/eval

wait_gpu_free() {
  local gpu_id=$1
  echo "[*] Waiting for GPU $gpu_id memory to be reclaimed..."
  for i in $(seq 1 30); do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo 0)
    if [ "$free_mem" -gt 35000 ]; then
      echo "[✓] GPU $gpu_id VRAM reclaimed: ${free_mem} MiB free."
      return 0
    fi
    sleep 2
  done
  echo "[!] Warning: GPU $gpu_id free memory is ${free_mem} MiB"
}

# -----------------------------------------------------------------------------
# 0. Ensure Attacker Lexi on GPU 1 (Port 8001) is running
# -----------------------------------------------------------------------------
if ! curl -s http://127.0.0.1:8001/v1/models | grep -q "data"; then
  echo "[*] Launching Attacker Model (Lexi-Uncensored-V2) on GPU 1 (Port 8001)..."
  CUDA_VISIBLE_DEVICES=1 nohup "$VLLM_BIN" serve "$LEXI_PATH" \
    --port 8001 --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enforce-eager < /dev/null > "$LOGS_DIR/vllm_attacker_8001.log" 2>&1 &
  for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8001/v1/models | grep -q "data"; then
      echo "[✓] Attacker Model is READY on port 8001!"
      break
    fi
    sleep 3
  done
else
  echo "[✓] Attacker Model is already running on port 8001."
fi

# -----------------------------------------------------------------------------
# 1. Evaluate InternLM-7B (internlm/internlm-chat-7b)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "[EVAL 3/4] internlm/internlm-chat-7b"
echo "=============================================================================="

pkill -9 -f "vllm serve.*8000" 2>/dev/null || true
wait_gpu_free 0

echo "[*] Launching InternLM-7B on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_BIN" serve "$INTERNLM_PATH" \
  --port 8000 --host 127.0.0.1 \
  --served-model-name internlm/internlm-chat-7b \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager < /dev/null > "$LOGS_DIR/vllm_internlm_8000.log" 2>&1 &

INTERNLM_PID=$!
echo "[*] InternLM-7B vLLM PID: $INTERNLM_PID. Waiting for readiness..."

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8000/v1/models | grep -q "internlm"; then
    echo "[✓] InternLM-7B is READY on port 8000!"
    break
  fi
  sleep 3
done

echo "[*] Pre-warming InternLM-7B..."
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"internlm/internlm-chat-7b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 8}" > /dev/null || true

echo "[*] Starting SAAGA evaluation for InternLM-7B..."
"$PYTHON_BIN" -m saaga evaluate \
  -m "$INTERNLM_PATH" \
  -p openai -u http://127.0.0.1:8000/v1 \
  --base-model-url http://127.0.0.1:8001/v1 \
  --base-model "$LEXI_PATH" \
  --base-model-provider openai \
  -d data/TensorTrust_subsets \
  --samples-per-tier 100 \
  --max-attempts 20 \
  --mode adaptive \
  --max-parallel 8 \
  --provider-workers 8 \
  --provider-timeout 120 \
  --seed 42 \
  -o results/eval/scn12_internlm7b | tee "$LOGS_DIR/eval_scn12_internlm7b.log"

echo "[✓] Finished InternLM-7B evaluation."
echo "[*] Shutting down InternLM-7B on GPU 0..."
kill "$INTERNLM_PID" 2>/dev/null || true
pkill -9 -f "vllm serve.*8000" 2>/dev/null || true

# -----------------------------------------------------------------------------
# 2. Evaluate Gemma-2B (google/gemma-2b-it)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "[EVAL 4/4] google/gemma-2b-it"
echo "=============================================================================="

wait_gpu_free 0

echo "[*] Launching Gemma-2B on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_BIN" serve "$GEMMA_PATH" \
  --port 8000 --host 127.0.0.1 \
  --served-model-name google/gemma-2b-it \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager < /dev/null > "$LOGS_DIR/vllm_gemma_8000.log" 2>&1 &

GEMMA_PID=$!
echo "[*] Gemma-2B vLLM PID: $GEMMA_PID. Waiting for readiness..."

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8000/v1/models | grep -q "gemma"; then
    echo "[✓] Gemma-2B is READY on port 8000!"
    break
  fi
  sleep 3
done

echo "[*] Pre-warming Gemma-2B..."
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"google/gemma-2b-it\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 8}" > /dev/null || true

echo "[*] Starting SAAGA evaluation for Gemma-2B..."
"$PYTHON_BIN" -m saaga evaluate \
  -m "$GEMMA_PATH" \
  -p openai -u http://127.0.0.1:8000/v1 \
  --base-model-url http://127.0.0.1:8001/v1 \
  --base-model "$LEXI_PATH" \
  --base-model-provider openai \
  -d data/TensorTrust_subsets \
  --samples-per-tier 100 \
  --max-attempts 20 \
  --mode adaptive \
  --max-parallel 8 \
  --provider-workers 8 \
  --provider-timeout 120 \
  --seed 42 \
  -o results/eval/scn12_gemma2b | tee "$LOGS_DIR/eval_scn12_gemma2b.log"

echo "[✓] Finished Gemma-2B evaluation."

# -----------------------------------------------------------------------------
# 3. Clean Shutdown of All Models
# -----------------------------------------------------------------------------
echo "[*] Cleaning up all server instances..."
kill "$GEMMA_PID" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true

echo ""
echo "=============================================================================="
echo "ALL 4 EVALUATIONS COMPLETE (Llama-3, Mistral-7B, InternLM-7B, Gemma-2B)!"
echo "End Time: $(date)"
echo "=============================================================================="
