from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
class QwenVlModel:
    def __init__(self, model_path: str):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[],
        )
        self.model = get_peft_model(self.model, self.lora_config)
        if hasattr(self.model, ""):
            


        
        