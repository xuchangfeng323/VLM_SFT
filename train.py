
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from transformers import AutoProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from utils import Arguments, resize_image
from model import QwenVlModel
from dataprocess import GroundedMNER
from metrics import (
    GroundedMNERMetric,
    parse_prediction,
    rescale_entities,
    write_jsonl,
)
from tracker import SwanLabTracker, build_tracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "args", "arg1.json")
# split 名 -> data_path 下对应的文件名
SPLIT_FILES = {
    "train": "train_sft.jsonl",
    "dev": "dev_sft.jsonl",
    "test": "test_sft.jsonl",
}

def strip_metric_prefix(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """去掉 val_/test_ 前缀，便于 swanlab 里按 split 分组展示。"""
    return {key[len(prefix):]: value for key, value in metrics.items() if key.startswith(prefix)}

def to_float(value: Any) -> Optional[float]:
    """把 tensor / 数值转成 float；转不了（比如 None）返回 None。"""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None

class Trainer:
    def __init__(self, args: Arguments, adapter_path: Optional[str] = None):
        self.args = args
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.grad_accum_steps,
            mixed_precision="bf16",
        )    
        self.model_config = QwenVlModel(args, adapter_path=adapter_path)
        self.model = self.model_config.get_model()
        self.processor = AutoProcessor.from_pretrained(args.model_dir)
        # 数据集按需构建：只跑测评时就不必去扫 train 的 7k 张图
        self.datasets: Dict[str, GroundedMNER] = {}
        # id -> (原始 dataloader, prepare 后的 dataloader)，避免重复 prepare
        self.prepared_loaders: Dict[int, Any] = {}
        self.model_prepared = False
        # swanlab 记录器：默认是空操作，train() 里按配置换成真的
        # （只用于监控训练；evaluate.py 单独测评时不会新建实验）
        self.tracker = SwanLabTracker()
        self.global_step = 0
        self.best_checkpoint: Optional[str] = None
        self.seed_everything(self.args.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.args.seed) 
    def seed_worker(self, worker_id: int):
        worker_seed = self.args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    def seed_everything(self, seed: Optional[int] = 42, *, verbose: bool = True):
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
        torch.use_deterministic_algorithms(False) 
    def pick_vision_inputs(self,batch: Dict[str, Any]) -> Dict[str, Any]:
        vision_inputs = {}
        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if k in batch:
                vision_inputs[k] = batch[k]
        return vision_inputs
    # ------------------------------ 数据 ------------------------------
    def get_dataset(self, split: str) -> GroundedMNER:
        """按 split 名（train/dev/test）懒加载数据集并缓存。"""
        if split not in self.datasets:
            if split not in SPLIT_FILES:
                raise KeyError(f"未知的 split: {split}，可选 {list(SPLIT_FILES)}")
            data_path = os.path.join(self.args.data_path, SPLIT_FILES[split])
            self.datasets[split] = GroundedMNER(
                self.args, data_path, self.args.image_root, self.processor
            )
        return self.datasets[split]
    def get_dataloader(self, split: str, shuffle: bool = False, batch_size: Optional[int] = None) -> DataLoader:
        return self.get_dataset(split).get_dataloader(
            batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            worker_init_fn=self.seed_worker,
            generator=self.generator,
        )
    def prepare_dataloader(self, dataloader):
        """同一份 dataloader 只 prepare 一次（已 prepare 过的直接复用）。"""
        entry = self.prepared_loaders.get(id(dataloader))
        if entry is not None and entry[0] is dataloader:
            return entry[1]
        prepared = self.accelerator.prepare(dataloader)
        self.prepared_loaders[id(dataloader)] = (dataloader, prepared)
        return prepared
    # ---------------------------- 优化器 ----------------------------
    def set_optimizer(self):
        if self.model is None:
            raise ValueError("Model is None")
        # 只对 LoRA 里维度 >= 2 的参数做权重衰减，bias / norm 之类不衰减
        decay_params = {
            n for n, p in self.model.named_parameters() if p.requires_grad and p.dim() >= 2
        }
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters() if n in decay_params and p.requires_grad],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if n not in decay_params and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.eps
        )
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.warmup_steps, self.total_steps)
    # ---------------------------- LoRA 存取 ----------------------------
    def save_checkpoint(self, save_dir: str, tag: str = "best") -> str:
        """保存当前 LoRA 权重（含 processor）：主进程落盘，其余进程等待，返回路径。"""
        path = os.path.join(save_dir, tag)
        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            self.accelerator.unwrap_model(self.model).save_pretrained(path)
            self.processor.save_pretrained(path)
        self.accelerator.wait_for_everyone()
        return path
    def load_adapter(self, adapter_path: str):
        """把某个 LoRA checkpoint 灌回当前模型（例如训练完用最优权重测测试集）。"""
        self.model_config.load_adapter(adapter_path)
    def close_tracker(self):
        """结束 swanlab 实验；没开启记录时是空操作，重复调用也没问题。"""
        self.tracker.finish()
    # ------------------------------ 训练 ------------------------------
    def train(self) -> Optional[str]:
        train_dataset = self.get_dataset("train")
        self.steps_per_epoch = max(
            1,
            math.ceil(math.ceil(len(train_dataset) / self.args.batch_size) / self.args.grad_accum_steps),
        )
        self.total_steps = self.steps_per_epoch * self.args.epochs
        self.warmup_steps = int(self.total_steps * self.args.warmup_ratio)
        self.set_optimizer()
        train_dataloader = self.get_dataloader("train", shuffle=True)
        val_dataloader = self.get_dataloader("dev", shuffle=False)
        self.model, self.optimizer, self.scheduler, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            train_dataloader,
            val_dataloader)
        self.model_prepared = True
        self.prepared_loaders[id(train_dataloader)] = (train_dataloader, train_dataloader)
        self.prepared_loaders[id(val_dataloader)] = (val_dataloader, val_dataloader)

        # ---- swanlab：只在主进程开一个实验，把超参和训练过程记进去 ----
        self.tracker = build_tracker(self.args, self.accelerator.is_main_process)
        self.tracker.log(
            {
                "train/num_samples": len(train_dataset),
                "train/num_steps": self.total_steps,
                "train/num_warmup_steps": self.warmup_steps,
                "train/grad_accum_steps": self.args.grad_accum_steps,
                "train/world_size": self.accelerator.num_processes,
            },
            step=0,
        )
        log_interval = max(1, int(getattr(self.args, "swanlab_log_interval", 10)))
        interval_loss, interval_steps = 0.0, 0
        best_score = float("-inf")
        bad_epochs = 0
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            epoch_loss, epoch_steps = 0.0, 0
            self.model.train()
            pbar = tqdm(
                train_dataloader,
                desc=f"epoch {epoch+1}/{self.args.epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in pbar:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    loss = self.model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        return_dict=False,
                        use_cache=False,
                        **self.pick_vision_inputs(batch),
                    )[0]
                    self.accelerator.backward(loss)
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    # 累积梯度时只有真正更新参数的那一步才算一个优化步
                    loss_value = loss.item()
                    grad_norm_value = to_float(grad_norm)
                    lr_value = self.scheduler.get_last_lr()[0]
                    interval_loss += loss_value
                    interval_steps += 1
                    epoch_loss += loss_value
                    epoch_steps += 1
                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        if self.global_step % log_interval == 0:
                            self.tracker.log(
                                {
                                    "train/loss": interval_loss / max(interval_steps, 1),
                                    "train/lr": lr_value,
                                    "train/grad_norm": grad_norm_value,
                                    "train/epoch": epoch + (self.global_step / max(self.steps_per_epoch, 1)),
                                },
                                step=self.global_step,
                            )
                            interval_loss, interval_steps = 0.0, 0
                    pbar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        lr=f"{lr_value:.2e}",
                    )
            # ---- 每个 epoch 在 dev 上测评，按 monitor（默认 val_f1）挑最好的 LoRA ----
            metrics = self.eval(val_dataloader, is_test=False, verbose=False)
            if self.args.monitor not in metrics and self.accelerator.is_main_process:
                print(f"[warn] 指标里没有 {self.args.monitor}，无法据此挑最优权重；"
                      f"可用的 key: {sorted(k for k in metrics if not k.endswith('per_type'))}")
            score = metrics.get(self.args.monitor, float("-inf"))
            improved = score > best_score
            if improved:
                best_score, bad_epochs = score, 0
                if self.args.save_dir:
                    self.best_checkpoint = self.save_checkpoint(self.args.save_dir)
            else:
                bad_epochs += 1
            # ---- swanlab：epoch 级别的曲线（train loss / val 指标 / 是否最优） ----
            self.tracker.log_metrics(
                strip_metric_prefix(metrics, "val_"), prefix="val/", step=self.global_step
            )
            self.tracker.log(
                {
                    "train/epoch_loss": epoch_loss / max(epoch_steps, 1),
                    "train/epoch_time_sec": time.time() - epoch_start,
                    "train/best_score": best_score,
                    "epoch": epoch + 1,
                },
                step=self.global_step,
            )
            if self.accelerator.is_main_process:
                print(f"[epoch {epoch+1}] {metrics['val_summary']} | "
                      f"loss={epoch_loss / max(epoch_steps, 1):.4f} "
                      f"time={time.time() - epoch_start:.1f}s"
                      f"{' | 保存最优 LoRA' if improved else ''}")
            if not improved and self.args.patience and bad_epochs >= self.args.patience:
                if self.accelerator.is_main_process:
                    print(f"{self.args.monitor} 连续 {bad_epochs} 个 epoch 没提升，提前停止")
                break
        if self.accelerator.is_main_process and self.best_checkpoint:
            print(f"best {self.args.monitor}={best_score:.4f} -> {self.best_checkpoint}")
        return self.best_checkpoint
    # ------------------------------ 测评 ------------------------------
    @torch.no_grad()
    def eval(
        self,
        dataloader,
        is_test: bool = False,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """在给定 dataloader 上跑一遍生成式测评，返回指标字典（前缀 val_ / test_）。

        流程：只把 prompt 送进模型生成实体串 -> 解析成实体三元组 -> 把预测框从
        "缩放后坐标系"换算回原图坐标系 -> 与金标一对一匹配 -> 统计 micro P/R/F1。

        多卡时用 gather_object 把各进程的样本汇总到每个进程，保证指标一致；
        预测文件只由主进程落盘。
        """
        split = "test" if is_test else "val"
        prefix = f"{split}_"  # 指标 key 的前缀，和配置里的 monitor（val_f1）对齐
        model = self.accelerator.unwrap_model(self.model)
        if not self.model_prepared:
            # 只做测评、没走过 accelerator.prepare 时，自己把模型搬到设备上
            self.model.to(self.accelerator.device)
        model.eval()
        dataloader = self.prepare_dataloader(dataloader)
        iou_threshold = getattr(self.args, "iou_threshold", 0.5)
        metric = GroundedMNERMetric(iou_threshold=iou_threshold)
        records: List[Dict[str, Any]] = []
        pbar = tqdm(
            dataloader, desc=split, disable=not self.accelerator.is_local_main_process
        )
        for batch in pbar:
            prompt_ids = batch["prompt_input_ids"]
            with self.accelerator.autocast():
                generated = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=batch["prompt_attention_mask"],
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    **self.pick_vision_inputs(batch),
                )
            # prompt 是左 padding，回复从 prompt 长度之后开始
            generated_ids = generated[:, prompt_ids.shape[1]:]
            pred_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            for i, pred_text in enumerate(pred_texts):
                orig_height, orig_width = batch["orig_sizes"][i]
                new_height, new_width = resize_image(
                    orig_height, orig_width, self.args.min_pixels, self.args.max_pixels
                )
                entities, malformed = parse_prediction(pred_text)
                # 模型是在 resize 后的图上生成框的，这里换算回原图坐标系再和金标比
                entities = rescale_entities(
                    entities,
                    scale_w=orig_width / float(new_width),
                    scale_h=orig_height / float(new_height),
                    width=orig_width,
                    height=orig_height,
                )
                records.append(metric.update(
                    gold_text=batch["gold_texts_orig"][i],
                    pred_entities=entities,
                    pred_raw=pred_text,
                    malformed=malformed,
                    meta={
                        "index": batch["indexes"][i],
                        "id": batch["ids"][i],
                        "image": batch["image_paths"][i],
                    },
                ))
            # 累计指标在每次 metric.update() 时就算好了，这里直接读来显示
            running = metric.running_metrics
            pbar.set_postfix(
                f1=f"{running['f1']:.3f}",
                text_f1=f"{running['text_f1']:.3f}",
                gold=metric.num_gold,
                pred=metric.num_pred,
            )

        # 多卡：每个进程都拿到全部样本的记录，保证指标一致；文件只由主进程写
        records = [r for r in gather_object(records) if r is not None]
        try:
            records.sort(key=lambda r: (r.get("index") is None, r.get("index")))
        except TypeError:
            pass
        if self.accelerator.num_processes > 1:
            metric = GroundedMNERMetric(iou_threshold=iou_threshold)
            for record in records:
                metric.update_record(record)
        metrics = metric.compute(prefix)
        metrics[f"{prefix}summary"] = metric.summary(prefix)
        if self.tracker.enabled and verbose:
            # 训练结束后的 test 测评也会记到同一个 swanlab 实验里
            self.tracker.log_metrics(
                strip_metric_prefix(metrics, prefix), prefix=f"{split}/", step=self.global_step
            )
        if self.accelerator.is_main_process:
            if verbose:
                print(f"[{split}] {metrics[f'{prefix}summary']}")
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                write_jsonl(records, save_path)
        return metrics
    def evaluate(self, splits: Sequence[str] = ("dev", "test"), save_dir: Optional[str] = None) -> Dict[str, Any]:
        """按 split 依次测评，并把预测与指标落盘（dev 的指标前缀是 val_，和 monitor 对齐）。"""
        results: Dict[str, Any] = {}
        for split in splits:
            dataloader = self.get_dataloader(
                split,
                shuffle=False,
                batch_size=getattr(self.args, "eval_batch_size", None) or self.args.batch_size,
            )
            save_path = os.path.join(save_dir, f"predictions_{split}.jsonl") if save_dir else None
            results.update(self.eval(dataloader, is_test=(split == "test"), save_path=save_path))
        if save_dir and self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"测评结果已保存到 {save_dir}")
        return results


def main():
    """用法：python train.py [train|eval] [config.json] [lora_checkpoint]

    train：训练 + 每个 epoch 在 dev 上测评挑最优 LoRA，最后用它测 test
    eval ：不训练，直接拿一个 LoRA checkpoint 测 dev/test
    """
    argv = list(sys.argv[1:])
    mode = "train"
    if argv and argv[0] in ("train", "eval"):
        mode = argv.pop(0)
    config_path = argv[0] if argv else DEFAULT_CONFIG
    checkpoint = argv[1] if len(argv) > 1 else None
    if mode == "eval" and not checkpoint:
        raise SystemExit("eval 模式需要指定 LoRA checkpoint：python train.py eval args/arg1.json checkpoint/best")

    args = Arguments(config_path)
    trainer: Optional[Trainer] = None
    try:
        if mode == "train":
            trainer = Trainer(args)
            best_checkpoint = trainer.train()
            if best_checkpoint:
                trainer.load_adapter(best_checkpoint)  # 用验证集挑出的最优权重测测试集
            splits = ("test",)
        else:
            trainer = Trainer(args, adapter_path=checkpoint)
            splits = ("dev", "test")
        # 预测和指标统一写到 <save_dir>/eval/，方便和保存的 LoRA 放在一起对比
        save_dir = os.path.join(args.save_dir, "eval") if args.save_dir else None
        trainer.evaluate(splits=splits, save_dir=save_dir)
    finally:
        # swanlab 实验在这里统一收尾（训练中途报错也会正常关闭）
        if trainer is not None:
            trainer.close_tracker()


if __name__ == "__main__":
    main()
