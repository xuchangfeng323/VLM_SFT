import argparse
import math
import os
import random
from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional
from dataprocess import TwitterGroundedMNERJsonl
import numpy as np
import torch
from accelerate import Accelerator
from collator import Qwen25VLSFTCollator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import parse_triples,extract_answer_text
from metrics import count_correct_mner,count_correct_eeg,count_correct_gmner,Metric_Result,prf
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
def seed_everything(seed: Optional[int] = 42, *, verbose: bool = True):

    #随机种子固定
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)  #torch CPU随机
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed) #GPU随机
    
    #deterministic_algorithms、 torch.backends.cudnn.benchmark如果设置为强制确定性 会导致训练异常缓慢
    #尽管这些加速可能会 多次训练的曲线/指标也可能有小幅波动，但对最终指标通常影响不大
    # 🚀 性能优先（关闭确定性）
    torch.backends.cudnn.benchmark = True #cuDNN 为当前输入形状自动搜索最快的卷积/算子实现。 若
    torch.backends.cudnn.deterministic = False  

    # 🚀 允许 TF32（Ampere+ 巨幅加速）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ❌ 不再使用 deterministic algorithms
    torch.use_deterministic_algorithms(False) #避免因“必须确定性”导致的慢速
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to the training jsonl file.")
    parser.add_argument(
        "--dev_jsonl",
        type=str,
        default=None,
        help="Optional dev jsonl used for validation after each epoch.",
    )

    parser.add_argument(
        "--test_jsonl",
        type=str,
        default=None,
        help="Optional test jsonl evaluated at the end of training.",
    )
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen25vl_7b_lora")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值（global norm）。<=0 表示关闭")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=r"^(model\.language_model.*\.(v_proj|q_proj|k_proj|down_proj|o_proj|up_proj|gate_proj))$",
        help='Regex (recommended, aligns with ms-swift). If you pass "all-linear", it will be mapped to the default regex.',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt used in both training and inference chat template. "
        "If None, a default prompt enforcing <think>/<answer> format is used.",
    )
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--dev_batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_iou", type=float, default=0.5)
    args = parser.parse_args()
    seed_everything(args.seed)
    def seed_worker(worker_id: int):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    is_cuda_device = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(args.seed) 
    accelerator = Accelerator( gradient_accumulation_steps=args.grad_accum_steps)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, dtype=args.dtype, trust_remote_code=True)
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    target = args.lora_target_modules
    if isinstance(target, str) and target.lower() in {"all-linear", "all_linear"}:
        target = r"^(model\.language_model.*\.(v_proj|q_proj|k_proj|down_proj|o_proj|up_proj|gate_proj))$"
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target,
    )
    model = get_peft_model(model, lora_cfg)
    model = accelerator.prepare(model)
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    collator = Qwen25VLSFTCollator(
        processor=processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        factor=28,
        system_prompt=args.system_prompt,
    )
    tokenizer = processor.tokenizer
    train_dataset = TwitterGroundedMNERJsonl(args.train_jsonl, image_root=args.image_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = accelerator.prepare(train_loader)
    if args.dev_jsonl is not None:
        dev_dataset = TwitterGroundedMNERJsonl(args.dev_jsonl, image_root=args.image_root)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.dev_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=is_cuda_device,
            collate_fn=collator,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(args.num_workers > 0),
        )
        dev_loader = accelerator.prepare(dev_loader)
        print(f"[debug] dev_samples={len(dev_dataset)}  dev_batch_size={args.dev_batch_size}  dev_batches={len(dev_loader)}")

    else:
        dev_dataset = None
    if args.test_jsonl is not None:
        test_dataset = TwitterGroundedMNERJsonl(args.test_jsonl, image_root=args.image_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=is_cuda_device,
            collate_fn=collator,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = accelerator.prepare(test_loader)
        print(f"[debug] test_samples={len(test_dataset)}  test_batch_size={args.test_batch_size}  test_batches={len(test_loader)}")

    else:
        test_dataset = None
    trainable_params = [p for p in model.parameters() if p.requires_grad]#可训练参数

    decay_params = [n for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_params and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_params and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    for epoch in range(args.epochs):
        model.train()
        if is_cuda_device:
            torch.cuda.empty_cache()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        disable=not accelerator.is_main_process
        )
        optimizer.zero_grad()
        for step, batch in progress_bar:
             with accelerator.accumulate(model):
                if batch is None:
                    continue
                vision_input = {}
                for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                    if k in batch:
                        vision_input[k] = batch[k]
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    **vision_input).loss
                accelerator.backward(loss)
                if args.max_grad_norm > 0.0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    progress_bar.set_postfix(
                        loss=f"{loss.detach().float().item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )   

        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        best_dev_gmner_f1=-1
        eeg_correct=0
        gmner_correct=0
        mnner_correct=0
        if dev_dataset is not None:
            model.eval()
            dev_progress_bar = tqdm(
            enumerate(dev_loader),
            total=len(dev_loader),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process
            )
            dev_loss = 0.0
            dev_num_samples = 0
            for dev_step, dev_batch in dev_progress_bar:
                if dev_batch is None:
                    continue
                with torch.inference_mode():
                    vision_input = {}
                    for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                        if k in dev_batch:
                            vision_input[k] = dev_batch[k]
                    generated_ids = model.generate(
                        input_ids=dev_batch["prompt_input_ids"],
                        attention_mask=dev_batch["prompt_attention_mask"],
                        **vision_input,
                        max_new_tokens=args.eval_max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        )
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(dev_batch["prompt_input_ids"], generated_ids)]     
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    gold_texts = dev_batch["gold_texts"]
                    
                    for pred_text, gold_text in zip(response, gold_texts):
                        pred_text = (pred_text or "").strip()
                        gold_text = (gold_text or "").strip()
                        gold_ans=extract_answer_text(gold_text)
                        pred_ans=extract_answer_text(pred_text)
                        bad_pred_bbox += sum(1 for t in pred_triples if not t.region_valid)
                        pred_triples=parse_triples(pred_ans)
                        gold_triples=parse_triples(gold_ans)
                        gmner_pred += len(pred_triples)
                        gmner_gold += len(gold_triples)
                        gmner_correct+=count_correct_gmner(pred_triples, gold_triples)
                        gmner_metric=prf(gmner_correct, gmner_pred, gmner_gold, "gmner")
                        mner_pred += len(pred_triples)
                        mner_gold += len(gold_triples)
                        mner_correct+=count_correct_mner(pred_triples, gold_triples)
                        mner_metric=prf(mner_correct, mner_pred, mner_gold, "mner")
                        eeg_pred += len(pred_triples)
                        eeg_gold += len(gold_triples)
                        eeg_correct+=count_correct_eeg(pred_triples, gold_triples)
                        eeg_metric=prf(eeg_correct, eeg_pred, eeg_gold, "eeg")
                        print("========== DEV GMNER (entity + type + region) ==========")
                        print(f"GMNER P: {gmner_metric.p:.4f},GMNER R: {gmner_metric.r:.4f},GMNER F1: {gmner_metric.f1:.4f}")
                        print("========== DEV MNER (entity + type) ==========")
                        print(f"MNER P: {mner_metric.p:.4f},MNER R: {mner_metric.r:.4f},MNER F1: {mner_metric.f1:.4f}")
                        print("========== DEV EEG (entity + type) ==========")
                        print(f"EEG P: {eeg_metric.p:.4f},EEG R: {eeg_metric.r:.4f},EEG F1: {eeg_metric.f1:.4f}")




