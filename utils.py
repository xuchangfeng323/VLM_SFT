import math
import os
import json
import re
ENTITY_SPLIT_RE = re.compile(r"[;；\n]+")
class Arguments:
    def __init__(self, args_path: str=""):
        self.args_dict = self._load_json_config(args_path)
        for key, value in self.args_dict.items():
            setattr(self, key, value)

    def _load_json_config(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
def resize_image(height, width, min_pixels, max_pixels,factor=28):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar < min_pixels:
        beta=math.sqrt(min_pixels/h_bar*w_bar)
        h_bar=math.ceil(h_bar*beta/factor)*factor
        w_bar=math.ceil(w_bar*beta/factor)*factor
    if h_bar * w_bar > max_pixels:
        beta=math.sqrt(h_bar*w_bar/max_pixels)
        h_bar=math.floor(h_bar*beta*factor)
        w_bar=math.floor(w_bar*beta*factor)
    return h_bar, w_bar
        
           
    

