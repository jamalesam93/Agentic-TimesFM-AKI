#!/usr/bin/env bash
# run_mimic_eval.sh — Evaluate fine-tuned DIKD model against MIMIC-IV real-world data
#
# This script:
#   1. Downloads the fine-tuned GGUF from Hugging Face
#   2. Boots a llama-server with the GGUF
#   3. Runs eval_dikd_batch.py against the MIMIC-IV holdout
#   4. Runs clinical tier gates and produces a comparison report
#
# Prerequisites:
#   - Vast.ai instance with GPU (A6000 / A100 recommended)
#   - This repo cloned to /root/AKI-training
#   - llama.cpp compiled at ~/llama.cpp  (or set LLAMA_CPP_PATH)
#   - pip install requests pandas
#   - huggingface-cli installed  (pip install huggingface_hub[cli])
#
# Usage:
#   bash scripts/run_mimic_eval.sh
#
# Environment variables:
#   HF_REPO          # HF repo with GGUF  (default: jamalesam93/gemma-4-12b-it-aki-sentinel)
#   GGUF_FILENAME    # GGUF file to download (default: dikd-gemma4-12b-q6_k.gguf)
#   LLAMA_CPP_PATH   # Path to llama.cpp   (default: ~/llama.cpp)
#   PORT             # Server port          (default: 1234)
#   SKIP_DOWNLOAD    # Set to 1 if GGUF is already local

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

HF_REPO="${HF_REPO:-jamalesam93/gemma-4-12b-it-aki-sentinel}"
GGUF_FILENAME="${GGUF_FILENAME:-dikd-gemma4-12b-q6_k.gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"
PORT="${PORT:-1234}"

GGUF_LOCAL="$PROJECT_DIR/exports/$GGUF_FILENAME"
MIMIC_EVAL_FILE="$PROJECT_DIR/data/mimic_eval_holdout.jsonl"
SYNTH_EVAL_FILE="$PROJECT_DIR/data/eval_holdout.jsonl"
MIMIC_OUT="$PROJECT_DIR/reports/mimic_eval_predictions.jsonl"
MIMIC_METRICS="$PROJECT_DIR/reports/mimic_eval_predictions.metrics.json"
SYNTH_OUT="$PROJECT_DIR/reports/synth_eval_predictions.jsonl"
SYNTH_METRICS="$PROJECT_DIR/reports/synth_eval_predictions.metrics.json"

echo "============================================================"
echo "  DIKD AKI SENTINEL — MIMIC-IV REAL-WORLD EVALUATION"
echo "============================================================"
echo "  Project     : $PROJECT_DIR"
echo "  HF Repo     : $HF_REPO"
echo "  GGUF File   : $GGUF_FILENAME"
echo "  MIMIC Eval  : $MIMIC_EVAL_FILE"
echo "  llama.cpp   : $LLAMA_CPP_DIR"
echo "============================================================"

# ── Stage 1: Download GGUF from Hugging Face ─────────────────
if [ "${SKIP_DOWNLOAD:-0}" -ne 1 ]; then
    echo -e "\n[Stage 1] Downloading GGUF from Hugging Face..."
    mkdir -p "$(dirname "$GGUF_LOCAL")"
    hf download "$HF_REPO" "$GGUF_FILENAME" \
        --local-dir "$(dirname "$GGUF_LOCAL")"
    echo "  ✓ Downloaded to $GGUF_LOCAL"
else
    echo -e "\n[Stage 1] SKIPPED download (SKIP_DOWNLOAD=1)"
fi

if [ ! -f "$GGUF_LOCAL" ]; then
    echo "Error: GGUF file not found at $GGUF_LOCAL"
    exit 1
fi

# ── Stage 2: Verify MIMIC-IV holdout exists ──────────────────
echo -e "\n[Stage 2] Checking evaluation files..."
if [ ! -f "$MIMIC_EVAL_FILE" ]; then
    echo "Error: MIMIC-IV holdout not found at $MIMIC_EVAL_FILE"
    echo "Run 'python scripts/build_mimic_holdout.py' first (requires mimic-iv-clinical-database-demo-2.2/)."
    exit 1
fi
MIMIC_LINES=$(wc -l < "$MIMIC_EVAL_FILE")
echo "  ✓ MIMIC-IV holdout: $MIMIC_LINES trajectories"

if [ -f "$SYNTH_EVAL_FILE" ]; then
    SYNTH_LINES=$(wc -l < "$SYNTH_EVAL_FILE")
    echo "  ✓ Synthetic holdout: $SYNTH_LINES trajectories (will also evaluate)"
    RUN_SYNTH=1
else
    echo "  ⓘ Synthetic holdout not found — skipping comparative evaluation"
    RUN_SYNTH=0
fi

# ── Stage 3: Boot llama-server ───────────────────────────────
echo -e "\n[Stage 3] Starting llama-server..."

SERVER_BIN=""
for bin_path in "$LLAMA_CPP_DIR/llama-server" "$LLAMA_CPP_DIR/build/bin/llama-server" "$LLAMA_CPP_DIR/build/bin/server"; do
    if [ -f "$bin_path" ] && [ -x "$bin_path" ]; then
        SERVER_BIN="$bin_path"
        break
    fi
done

if [ -z "$SERVER_BIN" ]; then
    echo "Error: llama-server binary not found in $LLAMA_CPP_DIR"
    echo "Build llama.cpp first:  cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)"
    exit 1
fi

echo "  → Launching: $SERVER_BIN --model $GGUF_LOCAL --port $PORT"
"$SERVER_BIN" --model "$GGUF_LOCAL" --port $PORT --host 127.0.0.1 --ctx-size 2048 --threads 8 --n-gpu-layers 999 > /tmp/llama-server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo -e "\n  → Shutting down llama-server (PID: $SERVER_PID)..."
    kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "  → Waiting for server to be ready..."
for i in $(seq 1 60); do
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" || echo "000")
    if [ "$STATUS_CODE" -eq 200 ]; then
        echo "  ✓ Server is ready!"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: llama-server crashed. Log:"
        tail -20 /tmp/llama-server.log
        exit 1
    fi
    if [ "$i" -eq 60 ]; then
        echo "Error: Server did not become ready in 120 seconds."
        tail -20 /tmp/llama-server.log
        exit 1
    fi
    sleep 2
done

# ── Stage 4: Run MIMIC-IV Evaluation ────────────────────────
echo -e "\n[Stage 4] Running MIMIC-IV real-world evaluation..."
mkdir -p "$PROJECT_DIR/reports"
python "$SCRIPT_DIR/eval_dikd_batch.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "dikd-gemma4-12b" \
    --data "$MIMIC_EVAL_FILE" \
    --out "$MIMIC_OUT" \
    --retry 2

echo -e "\n  → Running clinical tier gates on MIMIC-IV results..."
python "$SCRIPT_DIR/tier_gates.py" "$MIMIC_METRICS" --json "$PROJECT_DIR/reports/mimic_gate_results.json" || true

# ── Stage 5 (optional): Re-run Synthetic Evaluation ─────────
if [ "$RUN_SYNTH" -eq 1 ]; then
    echo -e "\n[Stage 5] Running synthetic holdout evaluation (for comparison)..."
    python "$SCRIPT_DIR/eval_dikd_batch.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --model "dikd-gemma4-12b" \
        --data "$SYNTH_EVAL_FILE" \
        --out "$SYNTH_OUT" \
        --retry 2

    echo -e "\n  → Running clinical tier gates on synthetic results..."
    python "$SCRIPT_DIR/tier_gates.py" "$SYNTH_METRICS" --json "$PROJECT_DIR/reports/synth_gate_results.json" || true
fi

# ── Stage 6: Comparison Report ──────────────────────────────
echo -e "\n[Stage 6] Generating comparison report..."
python "$SCRIPT_DIR/compare_eval_results.py" \
    --mimic-metrics "$MIMIC_METRICS" \
    --synth-metrics "${SYNTH_METRICS}" \
    --out "$PROJECT_DIR/reports/mimic_vs_synth_comparison.json" || true

echo -e "\n============================================================"
echo "  MIMIC-IV EVALUATION COMPLETE"
echo "============================================================"
echo "  Results:"
echo "    MIMIC-IV predictions : $MIMIC_OUT"
echo "    MIMIC-IV metrics     : $MIMIC_METRICS"
if [ "$RUN_SYNTH" -eq 1 ]; then
    echo "    Synth predictions    : $SYNTH_OUT"
    echo "    Synth metrics        : $SYNTH_METRICS"
fi
echo ""
echo "  Download results to your local machine:"
echo "    rsync -avzP -e 'ssh -p PORT' root@IP:/root/AKI-training/reports/ ./reports/"
echo "============================================================"
