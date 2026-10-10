from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import re
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[List[Box]]
    region_valid: bool = True

def _boxes_from_numbers(
    nums: list[float],
    *,
    strict: bool,
    where: str,
) -> list[Box]:
    if len(nums) % 4 != 0:
        raise ValueError(f"Expected 4 numbers for {where}, got {len(nums)}")
    if strict:
        raise ValueError(f"Strict mode is not supported for {where}")
        return []
    boxes: list[Box] = []
    for i in range(0, len(nums), 4):
        x1, y1, x2, y2 = nums[i:i + 4]
        boxes.append((x1, y1, x2, y2))
    return boxes
