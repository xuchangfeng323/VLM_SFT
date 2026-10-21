#!/usr/bin/env bash
# Convert: bash run_cot_to_sft.sh
# Override paths: INPUT_JSONL=... OUTPUT_JSONL=... bash run_cot_to_sft.sh
# Use reference labels: ANSWER_SOURCE=reference bash run_cot_to_sft.sh
# Python CLI flags can be appended, e.g. bash run_cot_to_sft.sh --overwrite
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN=${PYTHON_BIN:-python3}
INPUT_JSONL=${INPUT_JSONL:-data/cot/oldteacher_cot_train.jsonl}
OUTPUT_JSONL=${OUTPUT_JSONL:-data/sft/oldteacher_cot_train_sft.jsonl}
REFERENCE_SFT=${REFERENCE_SFT:-data/sft/train_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
ANSWER_SOURCE=${ANSWER_SOURCE:-cot}
ANSWER_FORMAT=${ANSWER_FORMAT:-tagged}

"$PYTHON_BIN" cot_to_sft.py \
  --input "$INPUT_JSONL" \
  --output "$OUTPUT_JSONL" \
  --reference-sft "$REFERENCE_SFT" \
  --image-root "$IMAGE_ROOT" \
  --check-images \
  --answer-source "$ANSWER_SOURCE" \
  --answer-format "$ANSWER_FORMAT" \
  "$@"
