#!/usr/bin/env bash
# run_vast_pipeline.sh — Complete DIKD End-to-End Training & Validation Pipeline
#
# Usage:
#   bash scripts/run_vast_pipeline.sh
#
# Environment variables (all default to 0 = enabled):
#   SKIP_TRAIN=1         # Skip QLoRA fine-tuning (reuse existing adapter)
#   SKIP_MERGE=1         # Skip adapter merge (reuse existing merged weights)
#   SKIP_GGUF=1          # Skip GGUF convert & quantization
#   SKIP_EVAL=1          # Skip batch evaluation & tier gates
#   LLAMA_CPP_PATH=~/llama.cpp   # Path to llama.cpp repository
#   SPLIT_SEED=42        # Seed for reproducible train/eval shuffle

set -euo pipefail

# ── Path Configuration ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MASTER_FILE="$PROJECT_DIR/output/llm_fine_tuning_dataset.jsonl"
TRAIN_FILE="$PROJECT_DIR/output/train_split.jsonl"
EVAL_FILE="$PROJECT_DIR/output/eval_split.jsonl"
OUTPUT_DIR="$PROJECT_DIR/outputs/dikd-gemma4-12b"
MERGED_DIR="$PROJECT_DIR/exports/dikd-gemma4-12b-merged-bf16"
GGUF_F16="$PROJECT_DIR/exports/dikd-gemma4-12b-f16.gguf"
GGUF_Q6_K="$PROJECT_DIR/exports/dikd-gemma4-12b-q6_k.gguf"
EVAL_OUT="$PROJECT_DIR/reports/eval_predictions.jsonl"
METRICS_FILE="$PROJECT_DIR/reports/eval_predictions.metrics.json"

LLAMA_CPP_DIR="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"
SPLIT_SEED="${SPLIT_SEED:-42}"

echo "=========================================================="
echo "      DIKD AKI SENTINEL - VAST.AI PIPELINE"
echo "=========================================================="
echo "Project Directory: $PROJECT_DIR"
echo "Llama.cpp Path   : $LLAMA_CPP_DIR"
echo "Skip flags       : TRAIN=${SKIP_TRAIN:-0} MERGE=${SKIP_MERGE:-0} GGUF=${SKIP_GGUF:-0} EVAL=${SKIP_EVAL:-0}"
echo "=========================================================="

# ── Helper: find a binary from a list of candidate paths ────────
find_binary() {
    local name="$1"; shift
    for bin_path in "$@"; do
        if [ -f "$bin_path" ] && [ -x "$bin_path" ]; then
            echo "$bin_path"
            return 0
        fi
    done
    echo "Error: $name not found in any of: $*" >&2
    return 1
}

# ── Pre-flight: master file must exist ──────────────────────────
if [ ! -f "$MASTER_FILE" ]; then
    echo "Error: Master dataset not found: $MASTER_FILE"
    echo "Run 'python main.py' first to generate the synthetic training data."
    exit 1
fi

# -------------------------------------------------------------------------
# Stage 1: Shuffled Train/Eval Split (80/20, seeded for reproducibility)
# -------------------------------------------------------------------------
echo -e "\n[Stage 1/7] Splitting dataset (80/20 shuffled, seed=$SPLIT_SEED)..."

python3 -c "
import json, random, sys

seed = int(sys.argv[1])
master = sys.argv[2]
train_out = sys.argv[3]
eval_out = sys.argv[4]

with open(master, 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]

random.seed(seed)
random.shuffle(lines)

split = int(len(lines) * 0.8)
if split <= 0: split = 1
if split >= len(lines): split = len(lines) - 1

with open(train_out, 'w', encoding='utf-8') as f:
    f.writelines(lines[:split])
with open(eval_out, 'w', encoding='utf-8') as f:
    f.writelines(lines[split:])

print(f'  Total: {len(lines)} | Train: {split} | Eval: {len(lines) - split}')
" "$SPLIT_SEED" "$MASTER_FILE" "$TRAIN_FILE" "$EVAL_FILE"

# -------------------------------------------------------------------------
# Stage 2: Dataset Validation (both train AND eval)
# -------------------------------------------------------------------------
echo -e "\n[Stage 2/7] Validating dataset schemas..."

echo "  -> Validating train split..."
python "$SCRIPT_DIR/validate_dikd_dataset.py" "$TRAIN_FILE"

echo "  -> Validating eval split..."
python "$SCRIPT_DIR/validate_dikd_dataset.py" "$EVAL_FILE"

# -------------------------------------------------------------------------
# Stage 3: Label Audit
# -------------------------------------------------------------------------
echo -e "\n[Stage 3/7] Auditing label balance..."
python "$SCRIPT_DIR/audit_labels.py" "$TRAIN_FILE" --json "$PROJECT_DIR/reports/audit_summary.json"

# -------------------------------------------------------------------------
# Stage 4: QLoRA Fine-Tuning
# -------------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" -ne 1 ]; then
    echo -e "\n[Stage 4/7] Starting QLoRA fine-tuning..."
    python "$SCRIPT_DIR/train_qlora_gemma4_12b.py" \
        --train-file "$TRAIN_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --backend unsloth
else
    echo -e "\n[Stage 4/7] SKIPPED (SKIP_TRAIN=1)"
fi

# -------------------------------------------------------------------------
# Stage 5: Merge Adapter
# -------------------------------------------------------------------------
if [ "${SKIP_MERGE:-0}" -ne 1 ]; then
    ADAPTER_DIR="$OUTPUT_DIR/lora_adapter"
    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "Error: Adapter not found at $ADAPTER_DIR"
        echo "Either run training first or point OUTPUT_DIR to an existing adapter."
        exit 1
    fi

    echo -e "\n[Stage 5/7] Merging adapter into base model..."
    python "$SCRIPT_DIR/merge_adapter.py" \
        --adapter-dir "$ADAPTER_DIR" \
        --out-dir "$MERGED_DIR"
else
    echo -e "\n[Stage 5/7] SKIPPED (SKIP_MERGE=1)"
fi

# -------------------------------------------------------------------------
# Stage 6: GGUF Convert & Quantization
# -------------------------------------------------------------------------
if [ "${SKIP_GGUF:-0}" -ne 1 ]; then
    echo -e "\n[Stage 6/7] Exporting to GGUF (convert + quantize)..."

    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        echo "Error: llama.cpp repository not found at $LLAMA_CPP_DIR"
        echo "Please clone it or set LLAMA_CPP_PATH."
        echo "Example: git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_DIR"
        exit 1
    fi

    # Find convert script
    CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        if [ -f "$LLAMA_CPP_DIR/convert.py" ]; then
            CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert.py"
        else
            echo "Error: convert_hf_to_gguf.py not found in $LLAMA_CPP_DIR"
            exit 1
        fi
    fi

    mkdir -p "$(dirname "$GGUF_F16")"

    echo "  -> Converting HF weights to f16 GGUF..."
    python "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16

    # Find quantize binary
    QUANTIZE_BIN=$(find_binary "llama-quantize" \
        "$LLAMA_CPP_DIR/llama-quantize" \
        "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
        "$LLAMA_CPP_DIR/build/bin/quantize")

    echo "  -> Quantizing GGUF to Q6_K..."
    "$QUANTIZE_BIN" "$GGUF_F16" "$GGUF_Q6_K" Q6_K

    echo "[SUCCESS] GGUF exported: $GGUF_Q6_K"
else
    echo -e "\n[Stage 6/7] SKIPPED (SKIP_GGUF=1)"
fi

# -------------------------------------------------------------------------
# Stage 7: Evaluation & Tier Gates  (fully independent of GGUF)
# -------------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" -ne 1 ]; then
    echo -e "\n[Stage 7/7] Starting batch evaluation..."

    # Verify quantized model exists
    if [ ! -f "$GGUF_Q6_K" ]; then
        echo "Error: Quantized model not found at $GGUF_Q6_K"
        echo "Run Stages 5-6 first, or set SKIP_EVAL=1 to skip."
        exit 1
    fi

    # Find server binary
    SERVER_BIN=$(find_binary "llama-server" \
        "$LLAMA_CPP_DIR/llama-server" \
        "$LLAMA_CPP_DIR/build/bin/llama-server" \
        "$LLAMA_CPP_DIR/build/bin/server")

    PORT=1234
    echo "  -> Booting llama-server on port $PORT..."

    "$SERVER_BIN" --model "$GGUF_Q6_K" --port $PORT --host 127.0.0.1 \
        --ctx-size 1024 --threads 8 > /tmp/llama-server.log 2>&1 &
    SERVER_PID=$!

    # Clean up server on any exit from this point
    cleanup() {
        echo "  -> Shutting down llama-server (PID: $SERVER_PID)..."
        kill -9 "$SERVER_PID" 2>/dev/null || true
    }
    trap cleanup EXIT

    # Wait for server readiness (up to 90 seconds)
    echo "  -> Waiting for server to initialize..."
    for i in $(seq 1 45); do
        STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "000")
        if [ "$STATUS_CODE" -eq 200 ]; then
            echo "  -> Server is ready!"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: llama-server crashed during startup. Log:"
            cat /tmp/llama-server.log
            exit 1
        fi
        if [ "$i" -eq 45 ]; then
            echo "Error: llama-server failed to become ready within 90 seconds."
            cat /tmp/llama-server.log
            exit 1
        fi
        sleep 2
    done

    echo "  -> Running batch evaluator..."
    python "$SCRIPT_DIR/eval_dikd_batch.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --model "dikd-gemma4-12b" \
        --data "$EVAL_FILE" \
        --out "$EVAL_OUT"

    echo "  -> Running clinical quality tier gates..."
    python "$SCRIPT_DIR/tier_gates.py" "$METRICS_FILE" --json "$PROJECT_DIR/reports/gate_results.json"

    echo -e "\n[SUCCESS] Evaluation complete. Check reports/gate_results.json"
else
    echo -e "\n[Stage 7/7] SKIPPED (SKIP_EVAL=1)"
fi

echo -e "\n=========================================================="
echo "      PIPELINE COMPLETE"
echo "=========================================================="
echo "Reports: $PROJECT_DIR/reports/"
echo "Adapter: $OUTPUT_DIR/lora_adapter/"
echo "=========================================================="
