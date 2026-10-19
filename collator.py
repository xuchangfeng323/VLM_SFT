from typing import Any, Dict, List, Optional, Tuple
from geometry import rescale_assistant_text_bboxes

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
    system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.processor = processor
        self.max_length = max_length
        self.factor = factor
        self.system_prompt = system_prompt
        if self.system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
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
        images: List[Any] = []
        scaled_assistant_texts: List[str] = []
        orig_assistant_texts: List[str] = []
        orig_sizes: List[Tuple[int, int]] = []
        valid_batch: List[Dict[str, Any]] = []
        for item in batch:
            image= item["image"]
            orig_w,orig_h=image.size
        
            try:
                scale_answer=rescale_assistant_text_bboxes(item["assistant_text"],
                orig_w=orig_w,
                orig_h=orig_h,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                factor=self.factor)
            except ValueError as exc:
                print(f"[WARNING] skip sample: {exc}")
                continue
            if scale_answer is None:
                continue
            valid_batch.append(item)
            images.append(image)
            scaled_assistant_texts.append(scale_answer)
            orig_assistant_texts.append(item["assistant_text"])
            orig_sizes.append((orig_w,orig_h))
        if not valid_batch:
            return None
        return {
            "images": images,
            "scaled_assistant_texts": scaled_assistant_texts,
            "orig_assistant_texts": orig_assistant_texts,
            "orig_sizes": orig_sizes,
            "valid_batch": valid_batch,
        }
    def _build_template_texts(
        self,
        prepared: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for item, scaled_answer in zip(
            prepared["valid_batch"],
            prepared["scaled_assistant_texts"],
        ):
            user_text = item["user_text"]
            prompt_messages = self.build_messages(
                user_text=user_text,
            )
            full_messages = self.build_messages(
                user_text=user_text,
                assistant_text=scaled_answer,
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
        return prompt_texts, full_texts
    
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
    def _encode_texts_and_images(
        self,
        prompt_texts: List[str],
        full_texts: List[str],
        images: List[Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        old_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "right"
            full_inputs = self.processor(
                text=full_texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = "left"
            prompt_inputs = self.processor(
                text=prompt_texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        return  full_inputs, prompt_inputs
    def _build_labels(
        self,
        full_inputs: Dict[str, Any],
        prompt_inputs: Dict[str, Any],
        ) -> Dict[str, Any]:
        input_ids=full_inputs["input_ids"]
        attention_mask=full_inputs["attention_mask"]
        labels=input_ids.clone()
        labels.fill_(-100)
        propmt_lens=prompt_inputs["attention_mask"].sum(dim=1)
        for i in range(input_ids.size(0)):
            prompt_len=int(propmt_lens[i].item())
            labels[i, prompt_len:] = input_ids[i, prompt_len:]
            labels[i, attention_mask[i] == 0] = -100
        full_inputs["labels"] = labels
        full_inputs["prompt_input_ids"] = prompt_inputs["input_ids"]
        full_inputs["prompt_attention_mask"] = prompt_inputs["attention_mask"]
        return full_inputs

            
    def __call__(
        self,
        batch: List[Dict[str, Any]],
    )->Optional[Dict[str, Any]]:
        prepared = self._prepare_valid_batch(batch)
        if prepared is None:
            return None
        prompt_texts, full_texts = self._build_template_texts(prepared)
        full_inputs, prompt_inputs = self._encode_texts_and_images(
        prompt_texts=prompt_texts,
        full_texts=full_texts,
        images=prepared["images"],
    )

        full_inputs=self._build_labels(full_inputs, prompt_inputs)
        
        full_inputs["gold_texts"]=prepared["scaled_assistant_texts"]
        full_inputs["gold_texts_orig"]=prepared["orig_assistant_texts"]
        full_inputs["orig_sizes"] = prepared["orig_sizes"]
        return full_inputs