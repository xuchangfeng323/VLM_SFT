from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import re
Box = Tuple[float, float, float, float]
FULLWIDTH_BAR_CANON = "|"

BAR_VARIANTS = [
    "｜",
    "│",
    "丨",
    "￨",
    "∣",
]

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
        
        if strict:
            raise ValueError(f"Expected 4 numbers for {where}, got {len(nums)}")
        return []
    boxes: list[Box] = []
    for i in range(0, len(nums), 4):
        x1, y1, x2, y2 = nums[i:i + 4]
        boxes.append((x1, y1, x2, y2))
    return boxes
def parse_bbox_regions(
    bbox_str:str,
    *,
    strict: bool,
    where: str)->tuple[Optional[list[Box]], bool]:
    if bbox_str is None:
        return None, False
    s=bbox_str.strip()
    if not s:
        if strict:
            raise ValueError(f"Empty bbox string for {where}")
        return None, True
    low=s.lower()
    if low in {"none", "n/a"}:
        return None, True
    try:
        parsed=json.loads(s)
        if (
            isinstance(parsed, list)
            and len(parsed) == 4
            and all(isinstance(v, (float,int)) for v in parsed) 
        ):
            return [
                Box(
                    float(parsed[0]),
                    float(parsed[1]),
                    float(parsed[2]),
                    float(parsed[3]),
                )
            ], True
        if(isinstance(parsed, list)
            and parsed
            and all(isinstance(box, list) for box in parsed)):
            boxes: list[Box] = []
            for box in parsed:
                if (
                    len(box) != 4
                    or not all(isinstance(v, (int, float)) for v in box)
                ):
                    raise ValueError(
                        f"[{where}] invalid box element: {box}"
                    )
                boxes.append(Box(
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ))
            return boxes, True
        if isinstance(parsed, list) and len(parsed) == 0:
                if strict:
                    raise ValueError(
                        f"[{where}] empty list bbox is invalid"
                    )

                return None, False
    except Exception:
        pass
    number_strings = re.findall(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        s,
    )
    if not number_strings:
        if strict:
            raise ValueError(
                f"[{where}] cannot parse bbox: {bbox_str!r}"
            )

        return None, False
    nums = [float(s) for s in number_strings]
    boxes = _boxes_from_numbers(
        nums=nums,
        strict=strict,
        where=where,
    )
    if not boxes:
        if strict:
            raise ValueError(
                f"[{where}] invalid bbox numbers: {bbox_str!r}"
            )

        return None, False
    return boxes, True
def normalize_separators(s: str) -> str:
    if s is None:
        return ""
    for bar in BAR_VARIANTS:
        s = s.replace(bar, FULLWIDTH_BAR_CANON)

    return s
def parse_triple_segment(
    seg: str,
    *,
    strict: bool,
    where: str,
) -> Optional[EntityTriple]:
    seg=seg.strip()
    if not seg or seg.lower() == "none":
        return None
    seg=normalize_separators(seg)
    parts = [
        part.strip()
        for part in seg.split("|")
        if part.strip()
    ]
    if len(parts) < 3:
        if strict:
            raise ValueError(
                f"[{where}] cannot parse segment "
                f"(too few parts): {seg!r}"
            )

        return None

        
