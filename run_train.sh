#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN=${PYTHON_BIN:-/home/xuchangfeng/miniforge3/envs/e2e-gmner/bin/python}
MODEL_NAME=${MODEL_NAME:-/home/model/Qwen2.5-VL-7B-Instruct}
TRAIN_JSONL=${TRAIN_JSONL:-/home/xuchangfeng/VLM_SFT/data/sft/pilot_teacher_sft.jsonl}
DEV_JSONL=${DEV_JSONL:-data/sft/dev_sft.jsonl}
TEST_JSONL=${TEST_JSONL:-data/sft/test_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/exp_7b_instruct_native}
GPUS=${GPUS:-0}
if [[ ! "$GPUS" =~ ^[0-9]+$ ]]; then
  printf '本训练入口使用单张 GPU；GPUS 请填写一个编号，例如 0。\n' >&2
  exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" -u train.py \
  --device cuda:0 \
  --model_name "$MODEL_NAME" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --test_jsonl "$TEST_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-8}" \
  --lr "${LR:-5e-4}" \
  --weight_decay 0.1 --warmup_ratio 0.03 \
  --max_length "${MAX_LENGTH:-2048}" \
  --bfloat16 --grad_ckpt \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --seed "${SEED:-42}" \
  --dev_batch_size "${DEV_BATCH_SIZE:-2}" \
  --test_batch_size "${TEST_BATCH_SIZE:-2}" \
  --eval_precision bf16 \
  --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS:-512}" \
  --eval_iou 0.5 --num_workers "${NUM_WORKERS:-4}" \
  --use_swanlab "$@"
