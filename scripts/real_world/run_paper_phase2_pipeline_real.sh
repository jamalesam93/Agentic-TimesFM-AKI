#!/usr/bin/env bash
# run_phd_phase2_pipeline.sh — Phase 2 LLM Fine-Tuning Pipeline for PAPER
#
# Usage:
#   bash scripts/run_phd_phase2_pipeline.sh
#
# Environment variables:
#   SKIP_TRAINING=1      # Skip actual QLoRA training and merge
#   SKIP_GGUF=1          # Skip GGUF convert & quantization
#   LLAMA_CPP_PATH=~/llama.cpp   # Path to llama.cpp repository

set -euo pipefail

# Path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Input datasets
TRAIN_FILE="$PROJECT_DIR/data/real_world/paper_sft_dataset.jsonl"
EVAL_FILE="$PROJECT_DIR/data/real_world/paper_eval_holdout.jsonl"

# Output destinations
OUTPUT_DIR="$PROJECT_DIR/outputs/real_world/paper-gemma-12b"
MERGED_DIR="$PROJECT_DIR/exports/real_world/paper-gemma-12b-merged-bf16"
GGUF_F16="$PROJECT_DIR/exports/real_world/paper-gemma-12b-f16.gguf"
GGUF_Q6_K="$PROJECT_DIR/exports/real_world/paper-gemma-12b-q6_k.gguf"
EVAL_OUT="$PROJECT_DIR/reports/real_world/paper_eval_predictions.jsonl"
METRICS_FILE="$PROJECT_DIR/reports/real_world/paper_eval_predictions.metrics.json"

LLAMA_CPP_DIR="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"

# Determine python command (Vast.ai/Ubuntu uses python3, Windows/Git Bash often uses python)
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "=========================================================="
echo "      PAPER PHASE 2 - LLM PIPELINE"
echo "=========================================================="
echo "Project Directory: $PROJECT_DIR"
echo "Llama.cpp Path   : $LLAMA_CPP_DIR"
echo "Python Executable: $PYTHON_BIN"
echo "=========================================================="

# -------------------------------------------------------------------------
# Stage 1: Generate Training Data
# -------------------------------------------------------------------------
if [ ! -f "$TRAIN_FILE" ]; then
    echo -e "\n[Stage 1] Training dataset missing. Generating..."
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_llm_dataset_real.py"
else
    echo -e "\n[Stage 1] Training dataset already exists: $TRAIN_FILE"
fi

# Generate an eval holdout if missing
if [ ! -f "$EVAL_FILE" ]; then
    echo -e "\n[Stage 1b] Holdout dataset missing. Generating..."
    # Reuse the build_holdout script but output to the PhD eval file
    "$PYTHON_BIN" "$SCRIPT_DIR/build_holdout_real.py" --n 200 --seed 9999 --out "$EVAL_FILE"
else
    echo -e "\n[Stage 1b] Holdout dataset already exists: $EVAL_FILE"
fi

# -------------------------------------------------------------------------
# Stage 2: Contamination Gate
# -------------------------------------------------------------------------
echo -e "\n[Stage 2] Running Contamination Check..."
"$PYTHON_BIN" "$SCRIPT_DIR/../check_contamination.py" "$TRAIN_FILE" "$EVAL_FILE"

# -------------------------------------------------------------------------
# Stage 3: Dataset Schema Validation
# -------------------------------------------------------------------------
echo -e "\n[Stage 3] Validating Dataset Schema..."
"$PYTHON_BIN" "$SCRIPT_DIR/../validate_dikd_dataset.py" "$TRAIN_FILE"

# -------------------------------------------------------------------------
# Stage 4: Structural Audit
# -------------------------------------------------------------------------
echo -e "\n[Stage 4] Auditing Label Balance..."
mkdir -p "$PROJECT_DIR/reports/real_world"
"$PYTHON_BIN" "$SCRIPT_DIR/../audit_labels.py" "$TRAIN_FILE" --json "$PROJECT_DIR/reports/real_world/paper_audit_summary.json"

# -------------------------------------------------------------------------
# Stage 5: Train QLoRA
# -------------------------------------------------------------------------
if [ "${SKIP_TRAINING:-0}" -ne 1 ]; then
    echo -e "\n[Stage 5] Starting QLoRA Fine-Tuning..."
    "$PYTHON_BIN" "$SCRIPT_DIR/../train_qlora_gemma4_12b.py" \
        --train-file "$TRAIN_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --backend unsloth
else
    echo -e "\n[Stage 5] SKIPPED: Fine-tuning training (SKIP_TRAINING=1)"
fi

# -------------------------------------------------------------------------
# Stage 6: Merge Adapter
# -------------------------------------------------------------------------
if [ "${SKIP_TRAINING:-0}" -ne 1 ]; then
    echo -e "\n[Stage 6] Merging Adapter into Base Model..."
    "$PYTHON_BIN" "$SCRIPT_DIR/../merge_adapter.py" \
        --adapter-dir "$OUTPUT_DIR/lora_adapter" \
        --out-dir "$MERGED_DIR"
else
    echo -e "\n[Stage 6] SKIPPED: Adapter merging (SKIP_TRAINING=1)"
fi

# -------------------------------------------------------------------------
# Stage 7: GGUF Convert & Quantization
# -------------------------------------------------------------------------
if [ "${SKIP_GGUF:-0}" -ne 1 ]; then
    echo -e "\n[Stage 7] Exporting to GGUF (convert + quantize)..."
    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        echo "Error: llama.cpp repository not found at $LLAMA_CPP_DIR"
        exit 1
    fi

    # Check for convert script
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
    "$PYTHON_BIN" "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16

    QUANTIZE_BIN=""
    for bin_path in "$LLAMA_CPP_DIR/llama-quantize" "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$LLAMA_CPP_DIR/build/bin/quantize"; do
        if [ -f "$bin_path" ] && [ -x "$bin_path" ]; then
            QUANTIZE_BIN="$bin_path"
            break
        fi
    done

    if [ -z "$QUANTIZE_BIN" ]; then
        echo "Error: llama-quantize binary not found."
        exit 1
    fi

    echo "  -> Quantizing GGUF to Q6_K..."
    "$QUANTIZE_BIN" "$GGUF_F16" "$GGUF_Q6_K" Q6_K

    echo "[SUCCESS] GGUF exported to $GGUF_Q6_K"
else
    echo -e "\n[Stage 7] SKIPPED: GGUF convert & quantization (SKIP_GGUF=1)"
fi

# -------------------------------------------------------------------------
# Stage 8: Evaluation & Tier Gates
# -------------------------------------------------------------------------
if [ "${SKIP_GGUF:-0}" -ne 1 ]; then
    echo -e "\n[Stage 8] Starting Batch Evaluation..."

    SERVER_BIN=""
    for bin_path in "$LLAMA_CPP_DIR/llama-server" "$LLAMA_CPP_DIR/build/bin/llama-server" "$LLAMA_CPP_DIR/build/bin/server"; do
        if [ -f "$bin_path" ] && [ -x "$bin_path" ]; then
            SERVER_BIN="$bin_path"
            break
        fi
    done

    if [ -z "$SERVER_BIN" ]; then
        echo "Error: llama-server binary not found."
        exit 1
    fi

    PORT=1235
    echo "  -> Booting llama-server on port $PORT with model: $GGUF_Q6_K"
    
    "$SERVER_BIN" --model "$GGUF_Q6_K" --port $PORT --host 127.0.0.1 --ctx-size 1024 --threads 8 > /tmp/llama-server-phd.log 2>&1 &
    SERVER_PID=$!

    cleanup() {
        echo "  -> Shutting down llama-server (PID: $SERVER_PID)..."
        kill -9 "$SERVER_PID" || true
    }
    trap cleanup EXIT

    echo "  -> Waiting for server to initialize..."
    for i in {1..45}; do
        STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" || echo "000")
        if [ "$STATUS_CODE" -eq 200 ]; then
            echo "  -> Server is ready!"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: llama-server crashed during startup."
            exit 1
        fi
        sleep 2
    done

    echo "  -> Running batch evaluator against llama-server..."
    "$PYTHON_BIN" "$SCRIPT_DIR/../eval_dikd_batch.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --model "paper-gemma-12b" \
        --data "$EVAL_FILE" \
        --out "$EVAL_OUT"

    echo "  -> Running clinical quality tier gates..."
    "$PYTHON_BIN" "$SCRIPT_DIR/../tier_gates.py" "$METRICS_FILE" --json "$PROJECT_DIR/reports/real_world/phd_gate_results.json"

    echo -e "\n[SUCCESS] Pipeline successfully completed!"
else
    echo -e "\n[Stage 8] SKIPPED: Batch evaluation and tier gates (SKIP_GGUF=1)"
fi
