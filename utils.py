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
ENTITY_SPLIT_RE = re.compile(r"[;；\n]+")
ANSWER_TAG_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.S | re.I,
)
@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[List[Box]]
    region_valid: bool = True
# 把一个float列表转换为Box列表
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
# 把bbox字符串解析为Box列表
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
# 把一个字符串解析为EntityTriple
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
    entity_name="|".join(parts[:-2])
    etype=parts[-2]
    bbox_str=parts[-1]
    if not entity_name or not etype:
        if strict:
            raise ValueError(
                f"[{where}] empty mention/type: {seg!r}"
            )

        return None
    regions, region_valid = parse_bbox_regions(
        bbox_str=bbox_str,
        strict=strict,
        where=where,
    )
    return EntityTriple(
        text=entity_name,
        region_valid=region_valid,
        etype=etype,
        regions=regions,
    )
# 把一个字符串解析为EntityTriple列表
def parse_triples(
    s: str,
    *,
    strict: bool,
    where: str,
) -> list[EntityTriple]:
    if s is None:
            return []
    s = normalize_separators(s).strip()
    if not s or s.lower() == "none":
        return []
    segments=[
        seg.strip()
        for seg in ENTITY_SPLIT_RE.split(s)
        if seg.strip()
    ]
    triples: list[EntityTriple] = []
    for seg in segments:
        if seg.lower() == "none":
            continue
        triple=parse_triple_segment(
            seg=seg,
            strict=strict,
            where=where,
        )
        if triple is not None:
            triples.append(triple)
    return triples
#从response中提取answer文本
def extract_answer_text(
    s: Optional[str],
) -> str:
    if not s:
        return ""
    s=s.strip()
    match=ANSWER_TAG_RE.search(s)
    if match:
        return (match.group(1) or "").strip()
    match_after_think = re.search(
        r"</think>\s*(.*)$",
        s,
        flags=re.S | re.I,
    )
    if match_after_think:
        return (match_after_think.group(1) or "").strip()
    return s
# 把EntityTriple列表转换为字符串格式
def triples_to_canon_text(
    triples: list[EntityTriple],
) -> str:
    if len(triples) == 0 or not triples:
        return ""
    triples_text_list=[]
    for triple in triples:
        bbox_str=""
        if triple.regions is  None:
            bbox_str="None"
        else:
            boxes=[]
            for x1, y1, x2, y2 in triple.regions:
                boxes.append(
                    [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ]
                )
            if len(boxes) == 1:
                bbox_str = "[" + ",".join(
                    map(str, boxes[0])
                ) + "]"
            else:
                bbox_str = json.dumps(
                    boxes,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        triples_text_list.append(
            f"{triple.text}|{triple.etype}|{bbox_str}"
        )
    return "; ".join(triples_text_list)

    
    


    


    
        
