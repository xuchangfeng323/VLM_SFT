import math
import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
Box = Tuple[int, int, int, int]
ENTITY_SPLIT_RE = re.compile(r"[;；\n]+")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[List[Box]]
    region_valid: bool = True

# 配置文件里容易写错的字段名 -> 代码里统一使用的名字（两边都会保留成属性）
ARG_ALIASES = {
    "lora_r": "r",
    "lora_target_modules": "target_modules",
    "warmup_radio": "warmup_ratio",
    "num_epochs": "epochs",
    "gradient_accumulation_steps": "grad_accum_steps",
}
# 配置文件里可以不写、缺省时用这些值兜底
ARG_DEFAULTS = {
    "seed": 42,
    "epochs": 3,
    "batch_size": 1,
    "grad_accum_steps": 1,
    "num_workers": 2,
    "lr": 2e-5,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "max_length": 1024,
    "max_new_tokens": 256,
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "monitor": "val_f1",
    "patience": 3,
    "save_dir": "checkpoint/",
    "data_path": "./data/sft/",
    "image_root": "",
    "iou_threshold": 0.5,
    # swanlab 实验记录（详细说明见 tracker.py）
    "use_swanlab": False,
    "swanlab_project": "VLM_SFT-GMNER",
    "swanlab_workspace": None,
    "swanlab_experiment_name": None,
    "swanlab_description": None,
    "swanlab_mode": None,  # None=云端 / "offline" / "local" / "disabled"
    "swanlab_logdir": None,
    "swanlab_log_interval": 10,
}


class Arguments:
    def __init__(self, args_path: str=""):
        config = self._load_json_config(args_path)
        for alias, name in ARG_ALIASES.items():
            if alias in config and name not in config:
                config[name] = config[alias]
        for key, value in ARG_DEFAULTS.items():
            config.setdefault(key, value)
        self.args_dict = config
        for key, value in config.items():
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
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))
def normalize_separators(s: str) -> str:
    if s is None:
        return ""
    for b in BAR_VARIANTS:
        s = s.replace(b, FULLWIDTH_BAR_CANON)
    return s
def parse_triples(s: str, *, strict: bool, where: str) -> List[EntityTriple]:
    if s is None:
        return []
    s = normalize_separators(s).strip()
    if not s or s.lower() == "none":
        return []
def extract_answer_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip()
    match_answer = ANSWER_TAG_RE.search(s)
    if match_answer:
        return (match_answer.group(1) or "").strip()
    match_answer2 = re.search(r"</think>\s*(.*)$", s, flags=re.S | re.I)
    if match_answer2:
        return (match_answer2.group(1) or "").strip()
    return s
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

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_INNER_BOX_SPLIT_RE = re.compile(r"\]\s*,\s*\[")
NONE_FIELD = "None"
def is_none_field(s: Optional[str]) -> bool:
    return s is None or s.strip().lower() in ("", "none")
def parse_regions(field: str) -> Optional[List[Box]]:
    """解析框字段，支持 'None'、'[x1,y1,x2,y2]' 与 '[[..],[..]]'。"""
    if is_none_field(field):
        return None
    s = field.strip().strip("[]")
    regions: List[Box] = []
    for chunk in _INNER_BOX_SPLIT_RE.split(s):
        nums = _NUM_RE.findall(chunk)
        if len(nums) != 4:
            raise ValueError(f"bad box field: {field!r}")
        regions.append(tuple(int(round(float(v))) for v in nums))
    if not regions:
        raise ValueError(f"bad box field: {field!r}")
    return regions
def format_regions(regions: Optional[List[Box]]) -> str:
    """输出与标注一致的格式：不可见为 None，可见为 [x1,y1,x2,y2] 或 [[..],[..]]。"""
    if not regions:
        return NONE_FIELD
    boxes = [",".join(str(int(v)) for v in box) for box in regions]
    if len(boxes) == 1:
        return f"[{boxes[0]}]"
    return "[" + ",".join(f"[{b}]" for b in boxes) + "]"
def _looks_like_triple(frag: str) -> bool:
    parts = frag.rsplit("|", 2)
    if len(parts) != 3:
        return False
    try:
        parse_regions(parts[2])
    except ValueError:
        return False
    return True
def split_entity_triples(s: Optional[str]) -> List[str]:
    """按 ';' 切分实体，并把实体名里 '&amp ;' 之类被误切开的片段合并回去。"""
    if not s:
        return []
    frags = re.split(r"[;；]", s)
    out: List[str] = []
    buf = ""
    for frag in frags:
        cand = frag if not buf else buf + ";" + frag
        if _looks_like_triple(cand):
            out.append(cand.strip())
            buf = ""
        else:
            buf = cand
    if buf.strip():
        out.append(buf.strip())
    return out
