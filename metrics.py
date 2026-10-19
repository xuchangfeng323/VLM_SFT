from utils import Box, EntityTriple
from typing import Tuple,Dict,List
import json
import re
from functools import lru_cache
@classmethod
@dataclass(frozen=True)
class Metric_Result:
    task: str
    p: float
    r: float
    f1: float
    
# 计算两个Box的IOU
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
#存在一个实体对应多个Box，计算最大IOU分数
def max_iou(
    pred_boxes: list[Box],
    gold_boxes: list[Box],
) -> float:
    best=0
    for pred_box in pred_boxes:
        for gold_box in gold_boxes:
            iou_score = iou(pred_box, gold_box)
            if iou_score > best:
                best = iou_score
    return best
# 计算EntityTriple的IOU分数是否超过阈值，二者都是 None 时返回 True，1.0，否则返回 False，0.0
def region_correct(
    p: EntityTriple,
    g: EntityTriple,
    iou_threshold: float,
) -> tuple[bool, float]:
    if not p.region_valid:
        return False, 0.0
    if p.regions is None and g.regions is None:
        return True, 1.0
    if p.regions is None or g.regions is None:
        return False, 0.0
    max_iou_score = max_iou(p.regions, g.regions)
    if max_iou_score >= iou_threshold:
        return True, max_iou_score
    return False, max_iou_score
# 统计正确匹配的EntityTriple数量
def count_correct_mner(
    preds: list[EntityTriple],
    golds: list[EntityTriple],
) -> int:
    correct=0
    used = [False] * len(golds)
    for pred in preds:
        for i, gold in enumerate(golds):
            if used[i]:
                continue
            if (
                pred.text == gold.text
                and pred.etype == gold.etype
            ):
                used[i] = True
                correct += 1
                break
    return correct
# 计算最佳二分匹配数量，不使用最大匹配算法
def _best_bipartite_match_count_by_score(
    num_preds: int,
    num_golds: int,
    edges: Dict[Tuple[int, int], float],
) -> int:
    @lru_cache(maxsize=None)
    def dp(
        i: int,
        used_mask: int,
    ) -> Tuple[int, float]:
        if i>=num_preds:
            return 0, 0.0
        best=dp(i+1, used_mask)
        for j in range(num_golds):
            if ((used_mask>>j)&1) != 0:
                continue
            score = edges.get((i, j), None)
            if score is None:
                continue
            candidate_count, candidate_score = dp(i+1, used_mask|(1<<j))
            candidate_count += 1
            candidate_score += score
            if candidate_count > best[0]:
                best = candidate_count, candidate_score
                
            elif (candidate_count == best[0] and candidate_score > best[1]):
                best = candidate_count, candidate_score
        return best
    return int(dp(0, 0)[0])
#按照 MNER 来统计 TP，MNER 只需要实体类型和文本匹配
def count_correct_gmner(
    pred_triples: List[EntityTriple],
    gold_triples: List[EntityTriple],
    iou_threshold: float = 0.5,
) -> int:
    edges: Dict[Tuple[int, int], float] = {}
    for i, pred in enumerate(pred_triples):
        for j, gold in enumerate(gold_triples):
            if pred.etype != gold.etype:
                continue
            if pred.text != gold.text:
                continue
            is_correct, score = region_correct(pred, gold, iou_threshold)
            if not is_correct:
                continue
            if is_correct:
                edges[(i, j)] = score
    return _best_bipartite_match_count_by_score(
        len(pred_triples),
        len(gold_triples),
        edges,
    )
# 计算P, R, F1
def prf(correct: int,
        predicted_len: int,
        gold_len: int,
        task: str,)->Metric_Result:
    p=correct/(predicted_len+1e-10)
    r=correct/(gold_len+1e-10)
    f=2*p*r/(p+r) if p+r > 0 else 0
    return Metric_Result(task=task, p=p, r=r, f1=f)

def count_correct_eeg(
    pred_triples: List[EntityTriple],
    gold_triples: List[EntityTriple],
    iou_threshold: float = 0.5,
) -> int:
    edges: Dict[Tuple[int, int], float] = {}
    for i, pred in enumerate(pred_triples):
        for j, gold in enumerate(gold_triples):
            if pred.text != gold.text:
                continue
            is_correct, score = region_correct(pred, gold, iou_threshold)
            if not is_correct:
                continue
            if is_correct:
                edges[(i, j)] = score
    correct = _best_bipartite_match_count_by_score(
        len(pred_triples),
        len(gold_triples),
        edges,
    )
    return correct
