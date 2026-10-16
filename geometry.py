import math
from typing import Tuple
def smart_resizes(
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
    return new_height,new_widt
    