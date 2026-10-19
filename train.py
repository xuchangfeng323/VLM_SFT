import argparse
import math
import os
import swanlab
import random
from datetime import datetime
from accelerate.tracking import SwanLabTracker
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
    parser.add_argument("--bfloat16", action="store_true")
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
    accelerator = Accelerator( gradient_accumulation_steps=args.grad_accum_steps)
    use_swanlab = args.use_swanlab
    if use_swanlab and accelerator.is_main_process:
        swanlab.init(
            project="qwen4ner",
            name=(
                f"{args.model_name.rstrip('/').split('/')[-1]}"
                f"-LoRA-r{args.lora_r}"
                f"-lr{args.lr}"
                f"-seed{args.seed}"

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
                "device": str(accelerator.device),
                "num_processes": accelerator.num_processes,
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
    if os.path.exists(args.output_dir):
        raise ValueError(f"Output directory {args.output_dir} already exists.")
    else:
        os.makedirs(args.output_dir )
    
    is_cuda_device = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(args.seed) 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, dtype=torch.bfloat16 if args.bfloat16 else None, trust_remote_code=True)
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
    best_dev_gmner_f1=-1
    early_stop_counter=0
    for epoch in range(args.epochs):
        model.train()
        
        log_path = os.path.join(args.output_dir, "log.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        best_checkpoint=os.path.join(args.output_dir,"best_model")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
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
                loss_for_log = loss.detach().float()
                if accelerator.sync_gradients:
                    loss_for_log = loss.detach().float()
                    mean_loss = accelerator.reduce(
                                    loss_for_log,
                                    reduction="mean",
                                ).item()
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            loss=f"{mean_loss:.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}"
                        )
                        if use_swanlab:
                            swanlab.log({
                                "loss": mean_loss,
                                "lr": scheduler.get_last_lr()[0],
                            })

        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        
        eeg_correct=0
        gmner_correct=0
        mner_correct=0
        pred_texts_list=[]
        gold_texts_list=[]
        if dev_dataset is not None:
            model.eval()
            dev_progress_bar = tqdm(
            enumerate(dev_loader),
            total=len(dev_loader),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process
            )
            record_response_dir=os.path.join(args.output_dir,f"dev_{epoch+1}.jsonl")
            bad_pred_bbox = 0
            gmner_correct = 0
            gmner_pred = 0
            gmner_gold = 0
            mner_correct = 0
            mner_pred = 0
            mner_gold = 0
            eeg_correct = 0
            eeg_pred = 0
            eeg_gold = 0
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
                    pred_texts_list.extend(response)
                    gold_texts_list.extend(gold_texts)
                    for pred_text, gold_text in zip(response, gold_texts):
                        pred_text = (pred_text or "").strip()
                        gold_text = (gold_text or "").strip()
                        gold_ans=extract_answer_text(gold_text)
                        pred_ans=extract_answer_text(pred_text)
                        pred_triples=parse_triples(pred_ans,strict=False,where="dev_pred")
                        gold_triples=parse_triples(gold_ans,strict=False,where="dev_gold")
                        reord_respone(response,gold_text,pred_triples,gold_triples,record_response_dir)
                        bad_pred_bbox += sum(1 for t in pred_triples if not t.region_valid)
                        gmner_pred += len(pred_triples)
                        gmner_gold += len(gold_triples)
                        gmner_correct+=count_correct_gmner(pred_triples, gold_triples,iou_threshold=args.eval_iou,)
                        mner_pred += len(pred_triples)
                        mner_gold += len(gold_triples)
                        mner_correct+=count_correct_mner(pred_triples, gold_triples)
                        eeg_pred += len(pred_triples)
                        eeg_gold += len(gold_triples)
                        eeg_correct+=count_correct_eeg(pred_triples, gold_triples,iou_threshold=args.eval_iou,)
        eeg_metric=prf(eeg_correct, eeg_pred, eeg_gold, "eeg")
        gmner_metric=prf(gmner_correct, gmner_pred, gmner_gold, "gmner")
        mner_metric=prf(mner_correct, mner_pred, mner_gold, "mner")
        print("========== DEV GMNER (entity + type + region) ==========")
        print(f"GMNER P: {gmner_metric.p:.4f},GMNER R: {gmner_metric.r:.4f},GMNER F1: {gmner_metric.f1:.4f}")
        print("========== DEV MNER (entity + type) ==========")
        print(f"MNER P: {mner_metric.p:.4f},MNER R: {mner_metric.r:.4f},MNER F1: {mner_metric.f1:.4f}")
        print("========== DEV EEG (entity + type) ==========")
        print(f"EEG P: {eeg_metric.p:.4f},EEG R: {eeg_metric.r:.4f},EEG F1: {eeg_metric.f1:.4f}")
        if accelerator.is_main_process:
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
                        }
            with open(log_dir, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        if best_dev_gmner_f1 < gmner_metric.f1:
            best_dev_gmner_f1=gmner_metric.f1
            early_stop_counter=0
            save_best_adapter(
                    best_checkpoint,
                    model,
                    processor,
                    accelerator,
                )
        else :
            early_stop_counter+=1
            if early_stop_counter>=args.patience:
                print("Early stop")
                break
            else:
                pass
    if test_dataset is not None:
        del model
        gc.collect()
        test_result=test(test_loader,model_dir,adapter_dir,out_dir,accelerator)
        with open(test_result_dir, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_result, ensure_ascii=False) + "\n")
        if use_swanlab :
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
        
def test(test_loader,model_dir,adapter_dir,out_dir,accelerator):
    if accelerator is None:
        accelerator=Accelerator()
    test_result_dir=os.path.join(out_dir,"test_result.jsonl")
    test_response_dir=os.path.join(out_dir,"test_response.jsonl")
    processor = AutoProcessor.from_pretrained(model_dir,trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir,trust_remote_code=True)
    model=PeftModel.from_pretrained(base_model,adapter_dir)
    model=accelerator.prepare(model)
    tokenizer=processor.tokenizer
    model.eval()
    with torch.inference_mode():
        test_progress_bar = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Test",
            disable=not accelerator.is_main_process
            )
        for test_batch in test_progress_bar:
            vision_input = {}
                    for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                        if k in test_batch:
                            vision_input[k] = test_batch[k]
                    generated_ids = model.generate(
                        input_ids=test_batch["prompt_input_ids"],
                        attention_mask=test_batch["prompt_attention_mask"],
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
                    pred_texts_list.extend(response)
                    gold_texts_list.extend(gold_texts)
                    for pred_text, gold_text in zip(response, gold_texts):
                        pred_text = (pred_text or "").strip()
                        gold_text = (gold_text or "").strip()
                        gold_ans=extract_answer_text(gold_text)
                        pred_ans=extract_answer_text(pred_text)
                        pred_triples=parse_triples(pred_ans,strict=False,where="test_pred")
                        gold_triples=parse_triples(gold_ans,strict=False,where="test_gold")
                        reord_respone(response,gold_text,pred_triples,gold_triples,test_response_dir)
                        bad_pred_bbox += sum(1 for t in pred_triples if not t.region_valid)
                        gmner_pred += len(pred_triples)
                        gmner_gold += len(gold_triples)
                        gmner_correct+=count_correct_gmner(pred_triples, gold_triples,iou_threshold=args.eval_iou)
                        mner_pred += len(pred_triples)
                        mner_gold += len(gold_triples)
                        mner_correct+=count_correct_mner(pred_triples, gold_triples)
                        eeg_pred += len(pred_triples)
                        eeg_gold += len(gold_triples)
                        eeg_correct+=count_correct_eeg(pred_triples, gold_triples,iou_threshold=args.eval_iou)
        eeg_metric=prf(eeg_correct, eeg_pred, eeg_gold, "eeg")
        gmner_metric=prf(gmner_correct, gmner_pred, gmner_gold, "gmner")
        mner_metric=prf(mner_correct, mner_pred, mner_gold, "mner")
        print("========== TEST GMNER (entity + type + region) ==========")
        print(f"GMNER P: {gmner_metric.p:.4f},GMNER R: {gmner_metric.r:.4f},GMNER F1: {gmner_metric.f1:.4f}")
        print("========== TEST MNER (entity + type) ==========")
        print(f"MNER P: {mner_metric.p:.4f},MNER R: {mner_metric.r:.4f},MNER F1: {mner_metric.f1:.4f}")
        print("========== TEST EEG (entity + type) ==========")
        print(f"EEG P: {eeg_metric.p:.4f},EEG R: {eeg_metric.r:.4f},EEG F1: {eeg_metric.f1:.4f}")
        return {
            "gmner":gmner_metric,
            "mner":mner_metric,
            "eeg":eeg_metric,
        }

            


    

    

    

def save_best_adapter(
    best_checkpoint_dir,
    model,
    processor,
    accelerator,
):
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(best_checkpoint_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    unwrapped_model.save_pretrained(
        best_checkpoint_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )

    if accelerator.is_main_process:
        processor.save_pretrained(best_checkpoint_dir)
        print(f"Best adapter saved to: {best_checkpoint_dir}")

    accelerator.wait_for_everyone()

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

