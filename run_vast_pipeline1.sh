#!/usr/bin/env bash
# Nassila Vast pipeline: validate → train → merge → GGUF → eval → reports
# v1.4: PHASE2_7_V1_4_WALKTHROUGH.md
# v1.5: PHASE2_8_V1_5_WALKTHROUGH.md
# v1.6: same walkthrough (decontaminated boost + weak/insufficient rows)
# v1.9: v1.8 + supported calibration + weak/supported prompt line
# v1.10: v1.8 base + v110 rebalance (drops v19 sup overdose)
#   PHASE=10 bash scripts/run_vast_pipeline.sh  # default
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TRAINING_DIR"

PHASE="${PHASE:-10}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
REPAIR="${REPAIR:-1}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"

case "$PHASE" in
  4a)
    OUTPUT_SUFFIX="v1.4a"
    REPORTS_PREFIX="v1_4a_"
    TRAIN_FILE="data/l3_grounding_train_v14a.jsonl"
    CHAT_FILE="data/l3_grounding_chat.jsonl"
    PREPARE_CMD=(python scripts/prepare_v14_train.py)
    AUDIT_JSON="reports/v1_4_audit_summary.json"
    SEQ_JSON="reports/v1_4_seq_audit.json"
    MODEL_CARD="MODEL_CARD_v1_4.md"
    ;;
  4b)
    OUTPUT_SUFFIX="v1.4b"
    REPORTS_PREFIX="v1_4b_"
    TRAIN_FILE="data/l3_grounding_train_v14a.jsonl"
    CHAT_FILE="data/l3_grounding_chat.jsonl"
    PREPARE_CMD=(python scripts/prepare_v14_train.py)
    AUDIT_JSON="reports/v1_4_audit_summary.json"
    SEQ_JSON="reports/v1_4_seq_audit.json"
    MODEL_CARD="MODEL_CARD_v1_4.md"
    ;;
  5)
    OUTPUT_SUFFIX="v1.5"
    REPORTS_PREFIX="v1_5_"
    TRAIN_FILE="data/l3_grounding_train_v15.jsonl"
    CHAT_FILE="data/l3_grounding_chat_v15.jsonl"
    PREPARE_CMD=(
      python scripts/prepare_v15_train.py
      --base data/l3_grounding_train_v14a.jsonl
      --boost data/l3_grounding_v15_boost.jsonl
      --out data/l3_grounding_train_v15.jsonl
    )
    AUDIT_JSON="reports/v1_5_audit_summary.json"
    SEQ_JSON="reports/v1_5_seq_audit.json"
    MODEL_CARD="EVAL_GONOGO.md"
    ;;
  6)
    OUTPUT_SUFFIX="v1.6"
    REPORTS_PREFIX="v1_6_"
    TRAIN_FILE="data/l3_grounding_train_v16.jsonl"
    CHAT_FILE="data/l3_grounding_chat_v16.jsonl"
    PREPARE_CMD=(
      python scripts/prepare_v15_train.py
      --base data/l3_grounding_train_v14a.jsonl
      --boost data/l3_grounding_v16_boost.jsonl
      --out data/l3_grounding_train_v16.jsonl
    )
    AUDIT_JSON="reports/v1_6_audit_summary.json"
    SEQ_JSON="reports/v1_6_seq_audit.json"
    MODEL_CARD="EVAL_GONOGO.md"
    ;;
  7)
    OUTPUT_SUFFIX="v1.7"
    REPORTS_PREFIX="v1_7_"
    TRAIN_FILE="data/l3_grounding_train_v17.jsonl"
    CHAT_FILE="data/l3_grounding_chat_v17.jsonl"
    PREPARE_CMD=(
      python scripts/prepare_v15_train.py
      --base data/l3_grounding_train_v14a.jsonl
      --boost data/l3_grounding_v16_boost.jsonl data/l3_grounding_v17_boost.jsonl
      --out data/l3_grounding_train_v17.jsonl
    )
    AUDIT_JSON="reports/v1_7_audit_summary.json"
    SEQ_JSON="reports/v1_7_seq_audit.json"
    MODEL_CARD="EVAL_GONOGO.md"
    ;;
  8)
    OUTPUT_SUFFIX="v1.8"
    REPORTS_PREFIX="v1_8_"
    TRAIN_FILE="data/l3_grounding_train_v18.jsonl"
    CHAT_FILE="data/l3_grounding_chat_v18.jsonl"
    PREPARE_CMD=(
      python scripts/prepare_v15_train.py
      --base data/l3_grounding_train_v14a.jsonl
      --boost data/l3_grounding_v16_boost.jsonl data/l3_grounding_v18_boost.jsonl
      --out data/l3_grounding_train_v18.jsonl
    )
    AUDIT_JSON="reports/v1_8_audit_summary.json"
    SEQ_JSON="reports/v1_8_seq_audit.json"
    MODEL_CARD="EVAL_GONOGO.md"
    ;;
  9)
    OUTPUT_SUFFIX="v1.9"
    REPORTS_PREFIX="v1_9_"
    TRAIN_FILE="data/l3_grounding_train_v19.jsonl"
    CHAT_FILE="data/l3_grounding_chat_v19.jsonl"
    PREPARE_CMD=(
      python scripts/prepare_v15_train.py
      --base data/l3_grounding_train_v14a.jsonl
      --boost data/l3_grounding_v16_boost.jsonl data/l3_grounding_v18_boost.jsonl data/l3_grounding_v19_boost.jsonl
      --out data/l3_grounding_train_v19.jsonl
    )
    AUDIT_JSON="reports/v1_9_audit_summary.json"
    SEQ_JSON="reports/v1_9_seq_audit.json"
    MODEL_CARD="EVAL_GONOGO.md"
    ;;
  10)
    OUTPUT_SUFFIX="v1.10"
    REPORTS_PREFIX="v1_10_"
    TRAIN_FILE="data/l3_grounding_train_v110.jsonl"
    CHAT_FILE="data/l3_grounding_chat_v110.jsonl"
    PREPARE_CMD=(
      python scripts/prepare_v15_train.py
      --base data/l3_grounding_train_v14a.jsonl
      --boost data/l3_grounding_v16_boost.jsonl data/l3_grounding_v18_boost.jsonl data/l3_grounding_v110_boost.jsonl
      --out data/l3_grounding_train_v110.jsonl
    )
    AUDIT_JSON="reports/v1_10_audit_summary.json"
    SEQ_JSON="reports/v1_10_seq_audit.json"
    MODEL_CARD="EVAL_GONOGO.md"
    ;;
  *)
    echo "PHASE must be 4a, 4b, 5, 6, 7, 8, 9, or 10 (got: $PHASE)" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="outputs/nassila-grounding-e4b-${OUTPUT_SUFFIX}"
MERGED_DIR="exports/hf-merged-${OUTPUT_SUFFIX}-bf16"
GGUF_F16="exports/nassila-grounding-e4b-${OUTPUT_SUFFIX}-f16.gguf"
GGUF_Q6="exports/nassila-grounding-e4b-${OUTPUT_SUFFIX}-q6_k.gguf"

echo "=== pipeline phase=${PHASE} (${OUTPUT_SUFFIX}) ==="

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "--- Build train file ---"
  "${PREPARE_CMD[@]}"
fi

echo "--- Contamination gate ---"
python scripts/check_contamination.py "$TRAIN_FILE"

echo "--- Validate train JSONL ---"
python scripts/validate_dataset.py "$TRAIN_FILE"

echo "--- Structural audit ---"
python scripts/audit_l3_labels.py "$TRAIN_FILE" --json "$AUDIT_JSON"

echo "--- Export chat + strict length ---"
python scripts/validate_dataset.py "$TRAIN_FILE" \
  --export-chat "$CHAT_FILE" \
  --strict-length 2048

echo "--- Sequence length audit ---"
python scripts/audit_chat_seq_lengths.py "$CHAT_FILE" --max-length 2048 \
  --json "$SEQ_JSON"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "--- Train QLoRA (phase ${PHASE}, save_strategy=no) ---"
  python scripts/train_qlora_gemma4_e4b.py \
    --train-file "$TRAIN_FILE" \
    --phase "$PHASE" \
    --chat-file "$CHAT_FILE"
fi

ADAPTER_DIR="${OUTPUT_DIR}/lora_adapter"
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "Adapter not found: $ADAPTER_DIR" >&2
  exit 1
fi

echo "--- Merge adapter ---"
python scripts/merge_adapter_gemma4.py \
  --adapter-dir "$ADAPTER_DIR" \
  --out-dir "$MERGED_DIR" \
  --max-seq-length 2048

echo "--- GGUF convert (F16) ---"
python "$HOME/llama.cpp/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$GGUF_F16" \
  --outtype f16

echo "--- GGUF quantize (Q6_K) ---"
"$LLAMA_BIN/llama-quantize" "$GGUF_F16" "$GGUF_Q6" Q6_K
ls -lh "$GGUF_Q6"

echo "--- Start llama-server (background) ---"
"$LLAMA_BIN/llama-server" \
  -m "$GGUF_Q6" \
  --host 127.0.0.1 \
  --port 1234 \
  --ctx-size 4096 &
SERVER_PID=$!
sleep 15

REPAIR_FLAG=""
if [[ "$REPAIR" == "1" ]]; then
  REPAIR_FLAG="--repair"
fi

echo "--- Batch eval (hardened 90-row holdout) ---"
python scripts/run_l3_eval_batch.py \
  --base-url http://127.0.0.1:1234 \
  --model nassila-grounding \
  --data data/eval_samples.jsonl data/eval_samples_extended.jsonl data/eval_holdout_90.jsonl \
  --chat-template --retry 1 $REPAIR_FLAG \
  --out "reports/${REPORTS_PREFIX}predictions.jsonl"

echo "--- Score reports ---"
python scripts/run_eval_reports.py \
  --predictions "reports/${REPORTS_PREFIX}predictions.jsonl" \
  --out-dir reports \
  --prefix "${REPORTS_PREFIX}" \
  $REPAIR_FLAG

echo "--- Cross-version failure matrix ---"
python scripts/compare_eval_versions.py --out reports/holdout_failure_matrix.md

kill "$SERVER_PID" 2>/dev/null || true

echo "=== Done. Reports: reports/${REPORTS_PREFIX}* ==="
echo "Check Tier 2 gates in ${REPORTS_PREFIX}eval_combined_report.json; update ${MODEL_CARD}"
