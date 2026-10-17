import math
from typing import Tuple, List
from utils import Box
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
)-> Tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(
            f"height and width must be at least {factor}, "
            f"got height={height}, width={width}"
        )
    image_ratio=max(height,width)/min(height,width)
    if image_ratio>200:raise ValueError(
            f"aspect ratio must be at most 200, got {image_ratio}"
        )
    new_height=round(height/factor)*factor
    new_width=round(width/factor)*factor
    if new_height*new_width<min_pixels:
        scale=math.sqrt(min_pixels/new_height/new_width)
        new_height=math.floor(height*scale/factor)*factor
        new_width=math.floor(width*scale/factor)*factor
    if new_height*new_width>max_pixels:
        scale=math.sqrt(max_pixels/new_height/new_width)
        new_height=math.floor(height*scale/factor)*factor
        new_width=math.floor(width*scale/factor)*factor
    return new_height,new_width
def _clamp_int(
    value: int,
    low: int,
    high: int,
) -> int:
    return max(low, min(value, high))
def scale_box_xyxy(
    box: Box,
    scale_w: float,
    scale_h: float,
    new_w: int,
    new_h: int,
) -> Tuple[int, int, int, int]:
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
def inverse_scale_box_xyxy(
    box: Box,
    scale_w: float,
    scale_h: float,
    orig_w: int,
    orig_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1o = int(round(x1 / scale_w))
    y1o = int(round(y1 / scale_h))
    x2o = int(round(x2 / scale_w))
    y2o = int(round(y2 / scale_h))
    x1o = _clamp_int(x1o, 0, orig_w)
    y1o = _clamp_int(y1o, 0, orig_h)
    x2o = _clamp_int(x2o, 0, orig_w)
    y2o = _clamp_int(y2o, 0, orig_h)
    if x1o > x2o:
        x1o, x2o = x2o, x1o

    if y1o > y2o:
        y1o, y2o = y2o, y1o
    return x1o, y1o, x2o, y2o