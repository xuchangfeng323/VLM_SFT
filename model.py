from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from typing import Optional
from utils import Arguments
class QwenVlModel:
    def __init__(self,args:Arguments,adapter_path:Optional[str]=None):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir)
        for param in self.model.parameters():
            param.requires_grad = False
        self.lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        )
        self.model = get_peft_model(self.model, self.lora_config)
        if adapter_path:
            # 测评/续训：把 checkpoint 里的 LoRA 权重灌回当前结构
            self.load_adapter(adapter_path)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
    def load_adapter(self,adapter_path:str):
        state_dict = load_peft_weights(adapter_path)
        result = set_peft_model_state_dict(self.model, state_dict)
        missing = getattr(result, "unexpected_keys", None)
        if missing:
            print(f"[warn] 未匹配上的权重 key: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        print(f"已加载 LoRA 权重: {adapter_path}")
    def get_model(self):
        return self.model

        
        


        
        
