from torch.utils.data import Dataset
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


        
    
    