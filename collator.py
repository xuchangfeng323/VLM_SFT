from typing import Any, Dict, List, Optional
DEFAULT_SYSTEM_PROMPT = (
    "You are an information extraction assistant for grounded "
    "multimodal named entity recognition. "
    "Given the image and text, first think briefly and then answer "
    "in a strict XML-like format. "
    "The final output must contain exactly one "
    "<think>...</think> block followed by exactly one "
    "<answer>...</answer> block. "
    "Inside <answer>, output only structured entity triples in the format "
    "mention|type|[x1,y1,x2,y2], separated by '; '. "
    "If there is no entity, output <answer>None</answer>. "
    "Do not add any extra explanation outside the tags."
)
class Qwen25VLSFTCollator:
    def __init__(self,
    processor,
    max_length=2048,
    *,
    min_pixels=None,
    max_pixels=None,
    factor=28,
    system_prompt=None):
        self.processor = processor
        self.max_length = max_length
        self.factor = factor
        self.system_prompt = system_prompt
        self.system_prompt=self.system_prompt.strip()
        image_processor = getattr(processor, "image_processor", None)
        self.min_pixels = int(
        min_pixels
        if min_pixels is not None
        else getattr(
            image_processor,
            "min_pixels",
            256 * 28 * 28,
        )
    )

        self.max_pixels = int(
            max_pixels
            if max_pixels is not None
            else getattr(
                image_processor,
                "max_pixels",
                1024 * 28 * 28,
            )
        )
        self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is not None:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.tokenizer.padding_side = "right"
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        self.system_prompt = system_prompt.strip()
    def _prepare_valid_batch(
            self,
            batch: List[Dict[str, Any]],
        ) -> Optional[Dict[str, Any]]:
        pass
    def build_messages(self,
        user_text:str,
        assistant_text:Optional[str] = None,
        ) -> List[Dict[str, Any]]:
        messages : List[Dict[str, Any]] = []
        if self.system_prompt :
            messages.append({"role": "system", "content": self.system_prompt})
        user_message = {"role": "user", 
        "content": [
            {"type":"image"},
            {"type":"text","text":user_text}
        ]}
        messages.append(user_message)
        if assistant_text is not None:
            assistant_message = {"role": "assistant", "content": assistant_text}
            messages.append(assistant_message)
        return messages
