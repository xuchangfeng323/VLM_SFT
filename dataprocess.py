from torch.utils.data import Dataset
from PIL import Image
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
        else:
            image = None
        

        

        
    
    