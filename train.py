
from cProfile import label
import random
from tqdm import tqdm

from transformers import AutoProcessor
import numpy as np
import torch
from typing import Optional, Dict, Any
import math
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from utils import Arguments
from model import QwenVlModel
from dataprocess import GroundedMNER

class Trainer:
    def __init__(self, args: Arguments):
        self.args = args
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.grad_accum_steps,
            mixed_precision="bf16",
        )    
        self.model_config = QwenVlModel(args)
        self.model = self.model_config.get_model()
        self.processor = AutoProcessor.from_pretrained(args.model_dir)
        self.train_dataset = GroundedMNER(args,args.data_path+"train_sft.jsonl", args.image_root)
        self.val_dataset = GroundedMNER(args,args.data_path+"dev_sft.jsonl", args.image_root)
        self.test_dataset = GroundedMNER(args,args.data_path+"test_sft.jsonl", args.image_root)
        self.steps_per_epoch = math.ceil(len(self.train_dataset) / args.grad_accum_steps)
        self.total_steps = self.steps_per_epoch * args.epochs
        self.warmup_steps = int(self.total_steps * args.warmup_ratio)
        self.seed_everything(self.args.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.args.seed) 
    def seed_worker(self, worker_id: int):
        worker_seed = self.args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    def set_optimizer(self):
        if self.model is None:
            raise ValueError("Model is None")
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters() if n in decay_params and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if n not in decay_params and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.warmup_steps, self.total_steps)
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
    def pick_vision_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
        vision_inputs = {}
        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if k in batch:
                vision_inputs[k] = batch[k]
        return vision_inputs
    def train(self):
        self.set_optimizer()
        self.train_dataloader=self.train_dataset.get_dataloader(
            self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers, 
            worker_init_fn=self.seed_worker, 
            generator=self.generator)
        self.val_dataloader=self.val_dataset.get_dataloader(
            self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            worker_init_fn=self.seed_worker, 
            generator=self.generator)
        self.test_dataloader=self.test_dataset.get_dataloader(
            self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            worker_init_fn=self.seed_worker, 
            generator=self.generator)
        self.model,self.optimizer,self.scheduler,self.train_dataloader,self.val_dataloader,self.test_dataloader=self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader)
        self.model.train()
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.train_dataloader, desc=f"epoch {epoch+1}/{args.epochs}")
            for step,batch in enumerate(pbar):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    loss=self.model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        return_dict=False,
                        use_cache=False,
                        **self.pick_vision_inputs(batch)
                                )[0]
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()


                
    

        
        
