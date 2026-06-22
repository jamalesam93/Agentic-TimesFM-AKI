#!/usr/bin/env bash
# run_vast_pipeline.sh — Complete DIKD End-to-End Training & Validation Pipeline
#
# Usage:
#   bash scripts/run_vast_pipeline.sh
#
# Environment variables:
#   SKIP_TRAINING=1      # Skip actual QLoRA training and merge (useful for testing eval)
#   SKIP_GGUF=1          # Skip GGUF convert & quantization (if llama.cpp is not installed)
#   LLAMA_CPP_PATH=~/llama.cpp   # Path to llama.cpp repository

set -euo pipefail

# Path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_FILE="$PROJECT_DIR/output/dikd_training_data_10k.jsonl"
EVAL_FILE="$PROJECT_DIR/data/eval_holdout.jsonl"
OUTPUT_DIR="$PROJECT_DIR/outputs/dikd-gemma4-12b"
MERGED_DIR="$PROJECT_DIR/exports/dikd-gemma4-12b-merged-bf16"
GGUF_F16="$PROJECT_DIR/exports/dikd-gemma4-12b-f16.gguf"
GGUF_Q6_K="$PROJECT_DIR/exports/dikd-gemma4-12b-q6_k.gguf"
EVAL_OUT="$PROJECT_DIR/reports/eval_predictions.jsonl"
METRICS_FILE="$PROJECT_DIR/reports/eval_predictions.metrics.json"

LLAMA_CPP_DIR="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"

echo "=========================================================="
echo "      DIKD AKI SENTINEL - VAST.AI PIPELINE"
echo "=========================================================="
echo "Project Directory: $PROJECT_DIR"
echo "Llama.cpp Path   : $LLAMA_CPP_DIR"
echo "=========================================================="

# -------------------------------------------------------------------------
# Stage 1: Build Holdout Set (if missing)
# -------------------------------------------------------------------------
if [ ! -f "$EVAL_FILE" ]; then
    echo -e "\n[Stage 1] Holdout dataset missing. Generating..."
    python "$SCRIPT_DIR/build_holdout.py" --n 200 --seed 9999 --out "$EVAL_FILE"
else
    echo -e "\n[Stage 1] Holdout dataset already exists: $EVAL_FILE"
fi

# -------------------------------------------------------------------------
# Stage 2: Contamination Gate
# -------------------------------------------------------------------------
echo -e "\n[Stage 2] Running Contamination Check..."
python "$SCRIPT_DIR/check_contamination.py" "$TRAIN_FILE" "$EVAL_FILE"

# -------------------------------------------------------------------------
# Stage 3: Dataset Schema Validation
# -------------------------------------------------------------------------
echo -e "\n[Stage 3] Validating Dataset Schema..."
python "$SCRIPT_DIR/validate_dikd_dataset.py" "$TRAIN_FILE"

# -------------------------------------------------------------------------
# Stage 4: Structural Audit
# -------------------------------------------------------------------------
echo -e "\n[Stage 4] Auditing Label Balance..."
python "$SCRIPT_DIR/audit_labels.py" "$TRAIN_FILE" --json "$PROJECT_DIR/reports/audit_summary.json"

# -------------------------------------------------------------------------
# Stage 5: Train QLoRA
# -------------------------------------------------------------------------
if [ "${SKIP_TRAINING:-0}" -ne 1 ]; then
    echo -e "\n[Stage 5] Starting QLoRA Fine-Tuning..."
    python "$SCRIPT_DIR/train_qlora_gemma4_12b.py" \
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
    python "$SCRIPT_DIR/merge_adapter.py" \
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
        echo "Please clone llama.cpp to $LLAMA_CPP_DIR or set LLAMA_CPP_PATH environment variable."
        echo "Example: git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_DIR"
        exit 1
    fi

    # Check for convert script
    CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        # Older llama.cpp versions might have convert.py
        if [ -f "$LLAMA_CPP_DIR/convert.py" ]; then
            CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert.py"
        else
            echo "Error: convert_hf_to_gguf.py or convert.py not found in $LLAMA_CPP_DIR"
            exit 1
        fi
    fi

    mkdir -p "$(dirname "$GGUF_F16")"
    
    echo "  -> Converting HF weights to f16 GGUF..."
    python "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16

    # Find quantize binary
    QUANTIZE_BIN=""
    for bin_path in "$LLAMA_CPP_DIR/llama-quantize" "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$LLAMA_CPP_DIR/build/bin/quantize"; do
        if [ -f "$bin_path" ] && [ -x "$bin_path" ]; then
            QUANTIZE_BIN="$bin_path"
            break
        fi
    done

    if [ -z "$QUANTIZE_BIN" ]; then
        echo "Error: llama-quantize binary not found in $LLAMA_CPP_DIR or its build directory."
        echo "Please build llama.cpp before running this stage."
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

    # Find server binary
    SERVER_BIN=""
    for bin_path in "$LLAMA_CPP_DIR/llama-server" "$LLAMA_CPP_DIR/build/bin/llama-server" "$LLAMA_CPP_DIR/build/bin/server"; do
        if [ -f "$bin_path" ] && [ -x "$bin_path" ]; then
            SERVER_BIN="$bin_path"
            break
        fi
    done

    if [ -z "$SERVER_BIN" ]; then
        echo "Error: llama-server binary not found. Cannot perform automated evaluation."
        exit 1
    fi

    PORT=1234
    echo "  -> Booting llama-server on port $PORT with model: $GGUF_Q6_K"
    
    # Spin up server in background
    "$SERVER_BIN" --model "$GGUF_Q6_K" --port $PORT --host 127.0.0.1 --ctx-size 1024 --threads 8 > /tmp/llama-server.log 2>&1 &
    SERVER_PID=$!

    # Function to kill server on script exit
    cleanup() {
        echo "  -> Shutting down llama-server (PID: $SERVER_PID)..."
        kill -9 "$SERVER_PID" || true
    }
    trap cleanup EXIT

    # Wait for server to be responsive
    echo "  -> Waiting for server to initialize..."
    for i in {1..30}; do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" || \
           curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" | grep -q "200" || \
           curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
            echo "  -> Server is ready!"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: llama-server crashed during startup. See /tmp/llama-server.log:"
            cat /tmp/llama-server.log
            exit 1
        fi
        sleep 2
    done

    echo "  -> Running batch evaluator against llama-server..."
    python "$SCRIPT_DIR/eval_dikd_batch.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --model "dikd-gemma4-12b" \
        --data "$EVAL_FILE" \
        --out "$EVAL_OUT"

    echo "  -> Running clinical quality tier gates..."
    python "$SCRIPT_DIR/tier_gates.py" "$METRICS_FILE" --json "$PROJECT_DIR/reports/gate_results.json"

    echo -e "\n[SUCCESS] Pipeline successfully completed!"
else
    echo -e "\n[Stage 8] SKIPPED: Batch evaluation and tier gates (SKIP_GGUF=1)"
fi
