#!/usr/bin/env bash
# =============================================================================
# SAAGA Evaluation: Google Gemma-2B-IT (4/4)
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

GEMMA_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
LEXI_PATH="/nlsasfs/home/isea/isea31/.cache/huggingface/hub/models--Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2/snapshots/f4617caeabd21f1820ac89bd125c80eda70901a7"

LOGS_DIR="$SAAGA_ROOT/logs"
mkdir -p "$LOGS_DIR" results/eval

echo "=============================================================================="
echo "[EVAL 4/4] Google Gemma-2B-IT Adaptive Evaluation"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# 1. Clean previous vLLM processes on ports 8000 & 8001
# -----------------------------------------------------------------------------
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

# -----------------------------------------------------------------------------
# 2. Launch Attacker Model (Lexi-Uncensored-V2) on GPU 1 (Port 8001)
# -----------------------------------------------------------------------------
echo "[*] Launching Attacker Model (Lexi-Uncensored-V2) on GPU 1 (Port 8001)..."
CUDA_VISIBLE_DEVICES=1 nohup "$VLLM_BIN" serve "$LEXI_PATH" \
  --port 8001 --host 127.0.0.1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager < /dev/null > "$LOGS_DIR/vllm_attacker_8001.log" 2>&1 &
ATTACKER_PID=$!
echo "[*] Attacker vLLM PID: $ATTACKER_PID"

# -----------------------------------------------------------------------------
# 3. Launch Target Victim Model (Gemma-2B-IT) on GPU 0 (Port 8000)
# -----------------------------------------------------------------------------
echo "[*] Launching Target Victim (Gemma-2B-IT) on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_BIN" serve "$GEMMA_PATH" \
  --port 8000 --host 127.0.0.1 \
  --served-model-name google/gemma-2b-it \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --enforce-eager < /dev/null > "$LOGS_DIR/vllm_gemma_8000.log" 2>&1 &
GEMMA_PID=$!
echo "[*] Gemma-2B vLLM PID: $GEMMA_PID"

# -----------------------------------------------------------------------------
# 4. Await Port 8001 Readiness
# -----------------------------------------------------------------------------
echo "[*] Waiting for Attacker Model on port 8001..."
while ! curl -s http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q "data"; do
  sleep 4
done
echo "[✓] Attacker Model is READY on port 8001!"

# -----------------------------------------------------------------------------
# 5. Await Port 8000 Readiness
# -----------------------------------------------------------------------------
echo "[*] Waiting for Gemma-2B on port 8000..."
while ! curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "gemma"; do
  sleep 4
done
echo "[✓] Gemma-2B is READY on port 8000!"

# -----------------------------------------------------------------------------
# 6. Pre-warm Gemma-2B and Lexi
# -----------------------------------------------------------------------------
echo "[*] Pre-warming models..."
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"google/gemma-2b-it\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 8}" > /dev/null || true

curl -s -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$LEXI_PATH\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 8}" > /dev/null || true
echo "[✓] Pre-warming complete!"

# -----------------------------------------------------------------------------
# 7. Start SAAGA Adaptive Evaluation
# -----------------------------------------------------------------------------
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

echo ""
echo "=============================================================================="
echo "[✓] Finished Gemma-2B evaluation!"
echo "End Time: $(date)"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# 8. Clean up servers
# -----------------------------------------------------------------------------
kill "$GEMMA_PID" "$ATTACKER_PID" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
