from utils import Box
from typing import Tuple
def iou(box1: Box, box2: Box) -> float:
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x2g - x1g) * max(0.0, y2g - y1g)
    ix1 = max(x1, x1g)
    iy1 = max(y1, y1g)
    ix2 = min(x2, x2g)
    iy2 = min(y2, y2g)
    area_inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if area_inter <= 0.0:
        return 0.0
    return area_inter / (area1 + area2 - area_inter)
