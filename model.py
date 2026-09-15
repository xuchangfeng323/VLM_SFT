from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from utils import Arguments
class QwenVlModel:
    def __init__(self,args:Arguments):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path)
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
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
    def get_model(self):
        return self.model


        
        