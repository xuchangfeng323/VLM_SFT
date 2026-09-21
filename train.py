
import random

import numpy as np
import torch
from typing import Optional

from transformers import get_cosine_schedule_with_warmup
from utils import Arguments
from model import QwenVlModel
from dataprocess import GroundedMNER

class Trainer:
    def __init__(self, args: Arguments):
        self.args = args
        self.model = QwenVlModel(args, dtype=torch.bfloat16)
        self.train_dataset = GroundedMNER(args.data_path+"train_sft.jsonl", args.image_root)
        self.val_dataset = GroundedMNER(args.data_path+"dev_sft.jsonl", args.image_root)
        self.test_dataset = GroundedMNER(args.data_path+"test_sft.jsonl", args.image_root)
        self.seed_everything()
        g = torch.Generator()
        g.manual_seed(args.seed) 
    def set_optimizer(self):
        if self.model is None:
            raise ValueError("Model is None")
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
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
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
    
    

        
        
