from PIL import Image
import json
import os
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Any
def normalize_user_text(s: str) -> str:
    return s.replace("<image> Text:", "<image>\nText:")


def strip_image_token_and_prefix(s: str) -> str:
    s = normalize_user_text(s)
    return s.replace("<image>", "").strip()
class TwitterGroundedMNERJsonl(Dataset):
    def __init__(self, 
                jsonl_path: str,
                image_root: Optional[str] = None,
                ):
        self.items: List[Dict[str, Any]] = []
        self.image_root = image_root
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.items.append(item)
    def __len__(self) -> int:
        return len(self.items)
    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.items[index]
        message = example["messages"]
        user_text = strip_image_token_and_prefix(message[0]["content"])
        assistant_text = message[-1]["content"]
        image_path=example["images"][0]
        if self.image_root and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        image=Image.open(image_path)
        return {
            "image": image,
            "user_text": user_text,
            "assistant_text": assistant_text,
               }
        
