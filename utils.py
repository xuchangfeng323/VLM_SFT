import math
import os
import json
import re
from dataclasses import dataclass
ENTITY_SPLIT_RE = re.compile(r"[;；\n]+")
@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[List[Box]]
    region_valid: bool = True
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
def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))
def scale_box(box,scale_w,scale_h,new_w,new_h): 
        x1, y1, x2, y2 = box
        x1n = int(round(x1 * scale_w))
        y1n = int(round(y1 * scale_h))
        x2n = int(round(x2 * scale_w))
        y2n = int(round(y2 * scale_h))
        x1n = _clamp_int(x1n, 0, new_w)
        y1n = _clamp_int(y1n, 0, new_h)
        x2n = _clamp_int(x2n, 0, new_w)
        y2n = _clamp_int(y2n, 0, new_h)
        if x1n > x2n:
            x1n, x2n = x2n, x1n
        if y1n > y2n:
            y1n, y2n = y2n, y1n
        return x1n, y1n, x2n, y2n
