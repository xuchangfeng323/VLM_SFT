import argparse
import math
import os
import swanlab
import random
from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional
from dataprocess import TwitterGroundedMNERJsonl
import numpy as np
import torch
from collator import Qwen25VLSFTCollator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import count_correct_mner,count_correct_eeg,count_correct_gmner,Metric_Result,prf
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
import gc
import json
from dataclasses import asdict
from peft import LoraConfig, PeftModel, get_peft_model
from utils import EntityTriple, parse_triples, extract_answer_text
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to the training jsonl file.")
    parser.add_argument(
        "--dev_jsonl",
        type=str,
        required=True,
        help="Required non-empty dev jsonl for global validation and best checkpoint selection.",
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
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_precision", choices=("bf16", "no"), default=None,
                        help="Evaluation autocast only; training autocast stays disabled.")
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
    parser.add_argument(
    "--use_swanlab",
    action="store_true",
    help="是否启用 SwanLab 实验记录")

    args = parser.parse_args()
    if args.epochs <= 0 or args.grad_accum_steps <= 0:
        parser.error("--epochs and --grad_accum_steps must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This training entry point uses one device; launch it with python train.py")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.eval_precision = args.eval_precision or ("bf16" if args.bfloat16 else "no")
    use_swanlab = args.use_swanlab
    if use_swanlab:
        swanlab.init(
            project="qwenvl4gmner",
            name=(
                f"{args.model_name.rstrip('/').split('/')[-1]}"
                f"-LoRA-r{args.lora_r}"
                f"-lr{args.lr}"
                f"-seed{args.seed}"+
                datetime.now().strftime("%Y%m%d%H%M%S")

            ),
            config={
                # 模型与训练
                "model": args.model_name,
                "method": "LoRA",
                "num_epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "max_grad_norm": args.max_grad_norm,
                "max_length": args.max_length,
                "grad_ckpt": args.grad_ckpt,
                "seed": args.seed,

                # 运行环境
                "device": str(device),
                "num_processes": 1,
                "training_autocast": False,
                "lora_dtype": "float32",
                "eval_precision": args.eval_precision,
                "num_workers": args.num_workers,

                # 数据与图片
                "train_jsonl": args.train_jsonl,
                "dev_jsonl": args.dev_jsonl,
                "test_jsonl": args.test_jsonl,
                "image_root": args.image_root,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,

                # LoRA
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,

                # 评估与输出
                "dev_batch_size": args.dev_batch_size,
                "test_batch_size": args.test_batch_size,
                "eval_max_new_tokens": args.eval_max_new_tokens,
                "eval_iou": args.eval_iou,
                "output_dir": args.output_dir,
            },
        )
    seed_everything(args.seed)
    def seed_worker(worker_id: int):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    log_path = os.path.join(args.output_dir, "log.jsonl")
    best_checkpoint = os.path.join(args.output_dir, "best_model")
    os.makedirs(args.output_dir, exist_ok=False)
    with open(log_path, "x", encoding="utf-8"):
        pass
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    is_cuda_device = device.type == "cuda"
    g = torch.Generator()
    g.manual_seed(args.seed) 
    processor = AutoProcessor.from_pretrained(
    args.model_name,
    min_pixels=args.min_pixels,
    max_pixels=args.max_pixels,
    trust_remote_code=True,
            )
    collator = Qwen25VLSFTCollator(
        processor=processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        factor=28,
        system_prompt=args.system_prompt,
    )
    train_collator = Qwen25VLSFTCollator(
        processor=processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        factor=28,
        system_prompt=args.system_prompt,
        skip_small_images=True,
    )
    tokenizer = processor.tokenizer
    train_dataset = TwitterGroundedMNERJsonl(args.train_jsonl, image_root=args.image_root)
    dev_dataset = TwitterGroundedMNERJsonl(args.dev_jsonl, image_root=args.image_root)
    test_dataset = (
        TwitterGroundedMNERJsonl(args.test_jsonl, image_root=args.image_root)
        if args.test_jsonl is not None else None
    )
    prevalidate_datasets(
        {"train": train_dataset, "dev": dev_dataset, "test": test_dataset},
        {"train": train_collator, "dev": collator, "test": collator},
        output_dir=args.output_dir,
    )
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=is_cuda_device,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        **{**loader_kwargs, "collate_fn": train_collator},
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, dtype=torch.bfloat16 if args.bfloat16 else None, trust_remote_code=True)
    model = model.to(device)
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
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=True)
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.print_trainable_parameters()
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]#可训练参数
    if not trainable_params or any(p.dtype != torch.float32 for p in trainable_params):
        raise RuntimeError("Training requires FP32 trainable LoRA weights")

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
    with open(os.path.join(args.output_dir, "runtime.json"), "w", encoding="utf-8") as f:
        json.dump({"torch": torch.__version__, "device": str(device),
                   "training_autocast": False, "lora_dtype": "torch.float32",
                   "eval_precision": args.eval_precision, "total_steps": total_steps,
                   "warmup_steps": warmup_steps,
                   "generation_config": model.generation_config.to_dict()}, f, indent=2)
    best_dev_gmner_f1=-1
    early_stop_counter=0
    global_step=0
    for epoch in range(args.epochs):
        model.train()
        
        if is_cuda_device:
            torch.cuda.empty_cache()
        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = False
        progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        )
        optimizer.zero_grad(set_to_none=True)
        accum_counter=0
        for step, batch in progress_bar:
            if batch is None:
                continue
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            del v
            vision_input = {}
            for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                if k in batch:
                    vision_input[k] = batch[k]
            if torch.is_autocast_enabled(device.type):
                raise RuntimeError("Training autocast must stay disabled")
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
                **vision_input)
            loss = out.loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, step={step}")
            loss_for_log = loss.detach().float() * args.grad_accum_steps
            mean_loss = loss_for_log.item()
            # Native backward: divide once, and count only non-empty microbatches.
            loss.backward()
            accum_counter += 1
            if accum_counter == args.grad_accum_steps:
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm,
                                                   error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_counter=0
                global_step+=1
                progress_bar.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )
                if use_swanlab:
                    swanlab.log({"loss": mean_loss, "lr": scheduler.get_last_lr()[0]})
            batch.clear()
            del out, loss, loss_for_log, batch, vision_input

        # Normalize the final short window before gradient clipping.
        if accum_counter > 0:
            for parameter in trainable_params:
                if parameter.grad is not None:
                    parameter.grad.mul_(args.grad_accum_steps / float(accum_counter))
            del parameter
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm,
                                               error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step+=1
            if use_swanlab:
                swanlab.log({"loss": mean_loss, "lr": scheduler.get_last_lr()[0]})

        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True
        
        dev_result = evaluate(
            args, dev_loader, model, tokenizer, device, "dev",
            response_path=os.path.join(args.output_dir, f"dev_{epoch + 1}.jsonl"),
        )
        gmner_metric = dev_result["gmner"]
        mner_metric = dev_result["mner"]
        eeg_metric = dev_result["eeg"]
        if use_swanlab :
            swanlab.log({
                "gmner_f1": gmner_metric.f1,
                "mner_f1": mner_metric.f1,
                "eeg_f1": eeg_metric.f1,
                            })
        log_record={
            "gmner_f1": gmner_metric.f1,
            "gmner_p": gmner_metric.p,
            "gmner_r": gmner_metric.r,
            "mner_f1": mner_metric.f1,
            "mner_p": mner_metric.p,
            "mner_r": mner_metric.r,
            "eeg_f1": eeg_metric.f1,
            "eeg_p": eeg_metric.p,
            "eeg_r": eeg_metric.r,
            "epoch": epoch + 1,
            "global_step": global_step,
                    }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        if best_dev_gmner_f1 < gmner_metric.f1:
            best_dev_gmner_f1=gmner_metric.f1
            early_stop_counter=0
            save_best_adapter(
                    best_checkpoint,
                    model,
                    processor,
                )
        else :
            early_stop_counter+=1
            if early_stop_counter>=args.patience:
                print("Early stop")
                break
            else:
                pass
    if test_dataset is not None:
        optimizer.zero_grad(set_to_none=True)
        # Release training owners before loading the saved best adapter for test.
        del optimizer, scheduler, trainable_params, optimizer_grouped_parameters
        del progress_bar, p
        del train_loader, dev_loader, model
        gc.collect()
        if is_cuda_device:
            torch.cuda.empty_cache()
        test_result = test(args, test_loader, args.model_name, best_checkpoint, args.output_dir, device)
        serializable_result = {name: asdict(metric) for name, metric in test_result.items()}
        with open(os.path.join(args.output_dir, "test_result.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable_result, ensure_ascii=False) + "\n")
        if use_swanlab:
            swanlab.log(
                {
                    "test_gmner_f1": test_result["gmner"].f1,
                    "test_mner_f1": test_result["mner"].f1,
                    "test_eeg_f1": test_result["eeg"].f1,
                    "test_gmner_p": test_result["gmner"].p,
                    "test_gmner_r": test_result["gmner"].r,
                    "test_mner_p": test_result["mner"].p,
                    "test_mner_r": test_result["mner"].r,
                    "test_eeg_p": test_result["eeg"].p,
                    "test_eeg_r": test_result["eeg"].r,
                }
            )
    if use_swanlab:
        swanlab.finish()
        
def prevalidate_datasets(datasets, collator, output_dir=None):
    """Validate samples and record small-image exclusions without changing sampler indices."""
    manifest = {}
    for split, dataset in datasets.items():
        if dataset is None:
            continue
        if len(dataset) == 0:
            raise ValueError(f"{split} dataset is empty")
        split_collator = collator[split] if isinstance(collator, dict) else collator
        excluded = []
        for index in tqdm(range(len(dataset)), desc=f"Validate {split}"):
            sample = None
            try:
                sample = dataset[index]
                sample["image"].load()
                if split_collator._prepare_valid_batch([sample]) is None:
                    item = dataset.items[index]
                    excluded.append({"index": index, "id": item.get("id", index),
                                     "image": item["images"][0],
                                     "size": list(sample["image"].size)})
            except Exception as exc:
                raise ValueError(f"{split} sample index {index}: {exc}") from exc
            finally:
                if sample is not None:
                    sample["image"].close()
        if len(excluded) == len(dataset):
            raise ValueError(f"{split} has no valid samples")
        manifest[split] = {"source_samples": len(dataset),
                           "effective_samples": len(dataset) - len(excluded), "excluded": excluded}
    if output_dir is not None:
        with open(os.path.join(output_dir, "data_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def evaluate(args, dataloader, model, tokenizer, device, split, response_path=None):
    """Compute the original micro-averaged metrics on one device."""
    model.eval()
    eval_model = model
    totals = torch.zeros(9, dtype=torch.long, device="cpu")
    sample_count = 0
    if response_path is not None:
        with open(response_path, "w", encoding="utf-8"):
            pass
    progress = tqdm(dataloader, total=len(dataloader), desc=split)
    with torch.inference_mode():
        for batch in progress:
            if batch is None:
                raise ValueError(f"{split}: empty batch; samples must not be skipped")
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            vision_inputs = {
                key: batch[key]
                for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
                if key in batch
            }
            # BF16 autocast is explicit and limited to evaluation.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and args.eval_precision == "bf16"):
                generated_ids = eval_model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    **vision_inputs,
                    max_new_tokens=args.eval_max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_width = batch["prompt_input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                generated_ids[:, prompt_width:], skip_special_tokens=True
            )
            gold_texts = batch["gold_texts"]
            if len(responses) != len(gold_texts) or not responses:
                raise ValueError(f"{split}: prediction/gold batch size mismatch or empty batch")
            rows = []
            records = []
            for response, gold_text in zip(responses, gold_texts):
                response = (response or "").strip()
                gold_text = (gold_text or "").strip()
                preds = parse_triples(extract_answer_text(response), strict=False, where=f"{split}_pred")
                golds = parse_triples(extract_answer_text(gold_text), strict=False, where=f"{split}_gold")
                num_pred, num_gold = len(preds), len(golds)
                rows.append([
                    count_correct_gmner(preds, golds, iou_threshold=args.eval_iou), num_pred, num_gold,
                    count_correct_mner(preds, golds), num_pred, num_gold,
                    count_correct_eeg(preds, golds, iou_threshold=args.eval_iou), num_pred, num_gold,
                ])
                if response_path is not None:
                    records.append({
                        "response": response,
                        "gold_texts": gold_text,
                        "pred_triples": [asdict(triple) for triple in preds],
                        "gold_triples": [asdict(triple) for triple in golds],
                    })
            counts = torch.tensor(rows, dtype=torch.long, device="cpu")
            totals += counts.sum(dim=0)
            sample_count += counts.shape[0]
            if response_path is not None:
                with open(response_path, "a", encoding="utf-8") as output:
                    for record in records:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
            batch.clear()
            del generated_ids, vision_inputs, batch
    if sample_count != len(dataloader.dataset):
        raise RuntimeError(
            f"{split}: evaluated {sample_count} samples, expected {len(dataloader.dataset)}"
        )
    values = totals.tolist()
    result = {
        task: prf(*values[offset:offset + 3], task)
        for task, offset in (("gmner", 0), ("mner", 3), ("eeg", 6))
    }
    for task, metric in result.items():
        print(f"{split} {task.upper()} P: {metric.p:.4f}, R: {metric.r:.4f}, F1: {metric.f1:.4f}")
    return result


def test(args, test_loader, model_dir, adapter_dir, out_dir, device):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16 if args.bfloat16 else None, trust_remote_code=True
    )
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(base_model, adapter_dir, autocast_adapter_dtype=True)
    return evaluate(
        args, test_loader, model, processor.tokenizer, device, "test",
        response_path=os.path.join(out_dir, "test_response.jsonl"),
    )


def save_best_adapter(
    best_checkpoint_dir,
    model,
    processor,
):
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    model.save_pretrained(
        best_checkpoint_dir,
        safe_serialization=True,
    )
    processor.save_pretrained(best_checkpoint_dir)
    print(f"Best adapter saved to: {best_checkpoint_dir}")

def reord_respone(
    response:str,
    gold_texts:str,
    pred_triples:list[EntityTriple],
    gold_triples:list[EntityTriple],
    dir,
):
    if dir is None:
        return
    record = {
    "response": response,
    "gold_texts": gold_texts,
    "pred_triples": [asdict(t) for t in pred_triples],
    "gold_triples": [asdict(t) for t in gold_triples],
        }
    with open(dir, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")
    

if __name__ == "__main__":
    main()

