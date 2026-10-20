#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN=${PYTHON_BIN:-/home/xuchangfeng/.conda/envs/e2e/bin/python}
MODEL_NAME=${MODEL_NAME:-/home/model/Qwen2.5-VL-3B-Instruct}
TRAIN_JSONL=${TRAIN_JSONL:-data/sft/train_sft.jsonl}
DEV_JSONL=${DEV_JSONL:-data/sft/dev_sft.jsonl}
TEST_JSONL=${TEST_JSONL:-data/sft/test_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/exp_3b_instruct}
GPUS=${GPUS:-0}  # nvidia-smi 中的 GPU 编号，例如 1 或 0,1
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

# 在 Accelerate 派生进程前检查，避免覆盖已有实验输出。
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  printf '错误：输出目录已存在：%s；请设置新的 OUTPUT_DIR。\n' "$OUTPUT_DIR" >&2
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NUM_PROCESSES=${#GPU_IDS[@]}
LAUNCH_ARGS=(--num_machines 1)
if ((NUM_PROCESSES > 1)); then
  LAUNCH_ARGS+=(--multi_gpu)
fi

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --gpu_ids 1\
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  "${LAUNCH_ARGS[@]}" \
  train.py \
  --model_name "$MODEL_NAME" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --test_jsonl "$TEST_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 5 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --lr 5e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --max_length 2048 \
  --bfloat16 \
  --grad_ckpt \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --dev_batch_size 2 \
  --test_batch_size 2 \
  --eval_max_new_tokens 512 \
  --eval_iou 0.5 \
  --num_workers 4 \
  --use_swanlab \
