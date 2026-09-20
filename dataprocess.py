from torch.utils.data import Dataset
from PIL import Image
import json
import os
from utils import EntityTriple, resize_image,scale_box

class GroundedMNER(Dataset):
    def __init__(self, data_path,image_root):
        self.items: List[Dict[str, Any]] = []
        self.image_root = image_root
        self.data_path = data_path
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.items.append(item)
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
    def collate_fn(self, batch,is_train=True):
        images = []
        scaled_assistant_texts: List[str] = []
        orig_assistant_texts: List[str] = []
        orig_sizes: List[Tuple[int, int]] = []
        user_messages: List[str] = []
        for x in batch:
            image = x["image"]
            images.append(image)
            height, width = image.size
            orig_sizes.append((height, width))
            h_new, w_new = resize_image(height, width, self.args.min_pixels, self.args.max_pixels)
            user_message=x['user_message']
            scale_w = w_new / float(width)
            scale_h = h_new / float(height)
            scaled_assistant_text_list=[]
            orign_assistant_message=x['assistant_message']
            entity_list=orign_assistant_message.split(";")
            orig_assistant_texts.append(orign_assistant_message)
            for i,entity in enumerate(entity_list):
                type,entity,box=entity.split("|")
                x1,y1,x2,y2=map(float,box.split(","))
                nx1,nx2,ny1,ny2=scale_box(x1,y1,x2,y2,scale_w,scale_h,w_new,h_new)
                scaled_assistant_text_list.append(f"{type}|{entity}|{nx1},{ny1},{nx2},{ny2}")
            if len(scaled_assistant_text_list)>0:
                scaled_assistant_texts.append(";".join(scaled_assistant_text_list))
            else:
                scaled_assistant_texts.append("")
            user_messages.append(user_message)
            
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
            

                
                
            
            


            

        
        
if __name__ == '__main__':
    
    dataset = GroundedMNER("data/sft/train_sft.jsonl", "data/sft/images")
    print(dataset[1])


        
    
    