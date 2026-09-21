from torch.utils.data import Dataset
from PIL import Image
import json
import os
from utils import Arguments
from utils import EntityTriple, resize_image, scale_box
from utils import parse_regions, format_regions, split_entity_triples, is_none_field
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple, Optional
class GroundedMNER(Dataset):
    def __init__(self, args:Arguments,data_path:str,image_root:str,processor:AutoProcessor):
        self.items: List[Dict[str, Any]] = []
        self.image_root = image_root
        self.data_path = data_path
        self.processor=processor
        self.args=args
        self.max_length = getattr(args, "max_length", 1024)
        self.system_prompt = getattr(args, "system_prompt", None)
        self.num_missing_images = 0
        self.dropped = {"missing_image": 0, "bad_image_size": 0, "bad_annotation": 0}
        self.tokenizer=processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        total = 0
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                total += 1
                if self._is_valid(item):
                    self.items.append(item)
        dropped = sum(self.dropped.values())
        if dropped:
            print(
                f"[{os.path.basename(self.data_path)}] 丢弃 {dropped}/{total} 个样本: {self.dropped}"
            )
    def _is_valid(self, item: Dict[str, Any]) -> bool:
        """不合格的样本直接丢弃：图找不到/读不了、尺寸或长宽比不满足 resize_image、标注解析不了。"""
        if self.image_root:
            image_path = os.path.join(self.image_root, item['images'][0])
            if not os.path.exists(image_path):
                self.dropped["missing_image"] += 1
                return False
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                self.dropped["bad_image_size"] += 1
                return False
        else:
            width, height = item.get('images_size', [0, 0])
        try:
            resize_image(height, width, self.args.min_pixels, self.args.max_pixels)
        except ValueError:
            self.dropped["bad_image_size"] += 1
            return False
        if not self._annotation_ok(item['messages'][1]['content']):
            self.dropped["bad_annotation"] += 1
            return False
        return True
    def _annotation_ok(self, assistant_message: str) -> bool:
        if is_none_field(assistant_message):
            return True
        triples = split_entity_triples(assistant_message)
        if not triples:
            return False
        # 切分结果必须能完整还原原始标注，否则说明有片段没被识别成三元组
        if ";".join(triples).replace(" ", "") != assistant_message.replace(" ", ""):
            return False
        for triple in triples:
            parts = triple.rsplit("|", 2)
            if len(parts) != 3:
                return False
            try:
                parse_regions(parts[2])
            except ValueError:
                return False
        return True
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        item = self.items[index]
        if self.image_root:
            image_path = os.path.join(self.image_root, item['images'][0])
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                image = None
                print(f"Image not found: {image_path}")   
        else:
            image = None
        message = item['messages']
        user_message = message[0]['content']
        user_message = user_message.replace("<image> Text:", "<image>\nText:")
        user_message = user_message.replace("<image>", "").strip()
        assistant_message = message[1]['content']
        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "image": image
        }
    def collate_fn(self, batch):
        images = []
        scaled_assistant_texts: List[str] = []
        orig_assistant_texts: List[str] = []
        orig_sizes: List[Tuple[int, int]] = []
        user_messages: List[str] = []
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for x in batch:
            image = x["image"]
            if image is None:
                self.num_missing_images += 1
                continue
            orig_width, orig_height = image.size
            orig_sizes.append((orig_height, orig_width))
            images.append(image)
            h_new, w_new = resize_image(orig_height, orig_width, self.args.min_pixels, self.args.max_pixels)
            user_message=x['user_message']
            # 框是原始图像坐标系，模型看到的是 resize_image 之后的尺寸，所以用原始尺寸换算
            scale_w = w_new / float(orig_width)
            scale_h = h_new / float(orig_height)
            scaled_assistant_text_list=[]
            orign_assistant_message=x['assistant_message']
            orig_assistant_texts.append(orign_assistant_message)
            if is_none_field(orign_assistant_message):
                scaled_assistant_text_list.append("None")
            else:
                for entity in split_entity_triples(orign_assistant_message):
                    entity_name, etype, box_field = entity.rsplit("|", 2)
                    regions = parse_regions(box_field)
                    if regions is None:
                        scaled_assistant_text_list.append(f"{entity_name}|{etype}|None")
                        continue
                    scaled_regions = [
                        scale_box(box, scale_w, scale_h, w_new, h_new) for box in regions
                    ]
                    scaled_assistant_text_list.append(
                        f"{entity_name}|{etype}|{format_regions(scaled_regions)}"
                    )
            if len(scaled_assistant_text_list)>0:
                scaled_assistant_texts.append(";".join(scaled_assistant_text_list))
            else:
                scaled_assistant_texts.append("")
            user_messages.append(user_message)
        if not images:
            raise ValueError(
                "batch 内所有样本的图像都加载失败，请检查 image_root 是否指向 jsonl 中 images 路径的父目录"
            )
        for scaled_assistant_text,user_message in zip(scaled_assistant_texts,user_messages):
            message=self.build_messages(user_message)
            full_message=self.build_messages(user_message,scaled_assistant_text)
            message_prompt = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            full_message_prompt = self.processor.apply_chat_template(
                full_message, tokenize=False, add_generation_prompt=False
            )
            prompt_texts.append(message_prompt)
            full_texts.append(full_message_prompt)
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            min_pixels=self.args.min_pixels,
            max_pixels=self.args.max_pixels,
            return_tensors="pt",
        )

        self.tokenizer.padding_side = "left"
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            min_pixels=self.args.min_pixels,
            max_pixels=self.args.max_pixels,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = old_side

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)

        labels = input_ids.clone()
        labels[:] = -100
        for i in range(input_ids.size(0)):
            p_len = int(prompt_lens[i].item())
            labels[i, p_len:] = input_ids[i, p_len:] 
            labels[i, attention_mask[i] == 0] = -100
        full_inputs["labels"] = labels
        full_inputs["prompt_input_ids"] = prompt_inputs["input_ids"]
        full_inputs["prompt_attention_mask"] = prompt_inputs["attention_mask"]
        full_inputs["gold_texts"] = scaled_assistant_texts     
        full_inputs["gold_texts_orig"] = orig_assistant_texts  
        full_inputs["orig_sizes"] = orig_sizes
        full_inputs["full_texts"]=full_texts
        return full_inputs

                
    def build_messages(self, user_text: str, assistant_text: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
        messages.append(user_msg)

        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages
    def get_dataloader(self,batch_size, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,collate_fn=self.collate_fn)           

                
                
            
            


            

        
        
if __name__ == '__main__':
    from transformers import Qwen2_5_VLForConditionalGeneration
    args=Arguments("/home/xuchangfeng/VLM_SFT/args/arg1.json")
    processor = AutoProcessor.from_pretrained("/home/model/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/home/model/Qwen2.5-VL-7B-Instruct")
    model.to("cuda")
    dataset = GroundedMNER(args,"data/sft/train_sft.jsonl", "data",processor)
    train_dataloader = dataset.get_dataloader(batch_size=1, shuffle=True)
    for batch in train_dataloader:
        prompt=batch["prompt_input_ids"].to("cuda")
        prompt_attention_mask=batch["prompt_attention_mask"].to("cuda")
        out = {}
        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if k in batch:
                out[k] = batch[k].to("cuda")
        outputs=model.generate(
            prompt,
            attention_mask=prompt_attention_mask,
            max_new_tokens=200,
            do_sample=False,
            **out
        )

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        # 只解码新生成的部分：prompt 是左 padding，前 prompt.shape[1] 列全是输入
        generated_ids = sequences[:, prompt.shape[1]:]
        outputs_texts=processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("model :", outputs_texts)
        print(batch["full_texts"])
        print(batch["orig_sizes"])
        
        break

    


        
    
    
