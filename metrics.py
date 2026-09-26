"""Grounded MNER 测评工具：解析模型输出 + 实体匹配 + P/R/F1 统计。

这个模块刻意不依赖 torch / transformers，只依赖 utils.py 里的解析函数，
所以既能直接用在训练/测评流程里，也能拿一份已保存的预测结果离线打分：

    python metrics.py --pred predictions_test.jsonl --gold data/sft/test_sft.jsonl

指标口径（沿用 GMNER 常见做法）：

    f1 / precision / recall     实体名 + 类型匹配，金标带框时预测也必须带框且 IoU >= 阈值
    text_*                      只比 实体名 + 类型，不看框
    grounded_*                  只看金标带框的子集，预测也必须带框且 IoU 达标

`f1` 是完整口径：金标没框（实体在图中不可见）时模型必须输出 None 才算对，
框错算错，多输出/漏输出分别记 FP / FN。

匹配用的是位掩码 DP（见 _match）：先最大化匹配数、再最大化总 IoU，而不是逐个
金标挑局部最优，避免"前面抢走了后面唯一能用的预测"这类次优结果。状态数是
O(num_pred × 2^num_gold)，只和单个样本的实体数有关（这批数据最多 6 个）。
"""

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils import (
    Box,
    EntityTriple,
    extract_answer_text,
    format_regions,
    is_none_field,
    parse_regions,
    scale_box,
    split_entity_triples,
)

DEFAULT_IOU_THRESHOLD = 0.5

_WS_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_NONE_TOKENS = {"none", "null", "无", "no entity", "no entities", "no entity found"}


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def normalize_name(s: Optional[str]) -> str:
    """实体名归一化：小写 + 压掉多余空白 + 去掉首尾标点。"""
    s = _WS_RE.sub(" ", (s or "").strip().lower())
    return s.strip(" \t\"'`.,;:|-")


def normalize_type(s: Optional[str]) -> str:
    return _WS_RE.sub("", (s or "").strip().lower())


def entity_key(entity: EntityTriple) -> Tuple[str, str]:
    return normalize_name(entity.text), normalize_type(entity.etype)


def parse_annotations(text: Optional[str]) -> Tuple[List[EntityTriple], List[str]]:
    """解析 `entity|type|box` 标注串（金标和规范化后的预测都用这套）。"""
    if is_none_field(text):
        return [], []
    entities: List[EntityTriple] = []
    malformed: List[str] = []
    for frag in split_entity_triples(text):
        name, etype, box_field = _split_triple(frag)
        if name is None:
            malformed.append(frag)
            continue
        regions = None
        if box_field is not None and not is_none_field(box_field):
            try:
                regions = parse_regions(box_field)
            except ValueError:
                malformed.append(frag)
        entities.append(EntityTriple(name, etype, regions))
    return entities, malformed


def parse_prediction(raw: Optional[str]) -> Tuple[List[EntityTriple], List[str]]:
    """解析模型生成的一段文本，返回 (entities, malformed_fragments)。

    容忍常见脏输出：markdown 代码块、<answer></answer> 包裹、think 前缀、
    缺字段（`name|type` 当作不带框）、框里多余空格/换行等。
    解析不了的片段不静默丢弃，而是放进 malformed 用来统计格式错误率。
    """
    text = _WS_RE.sub(" ", (raw or "").replace("\n", " ")).strip()
    text = _FENCE_RE.sub("", text).strip()
    text = extract_answer_text(text)  # 去掉 <answer> / 思考过程前缀
    text = _FENCE_RE.sub("", text).strip().strip("\"'").strip()
    if not text or text.lower() in _NONE_TOKENS:
        return [], []
    if text.startswith("{"):
        return _parse_json_prediction(text)
    entities: List[EntityTriple] = []
    malformed: List[str] = []
    for frag in split_entity_triples(text):
        if len(frag.rsplit("|", 2)) < 2:
            malformed.append(frag)  # 缺类型字段，格式不完整
        name, etype, box_field = _split_triple(frag)
        if name is None:
            malformed.append(frag)
            continue
        regions = None
        if box_field is not None and not is_none_field(box_field):
            try:
                regions = parse_regions(box_field)
            except ValueError:
                malformed.append(frag)
        entities.append(EntityTriple(name, etype, regions))
    return [e for e in entities if normalize_name(e.text)], malformed


def _split_triple(frag: str) -> Tuple[Optional[str], str, Optional[str]]:
    """把 `entity|type|box` 拆成三段，容忍缺字段。"""
    frag = frag.strip().strip(",").strip()
    if not frag:
        return None, "", None
    parts = frag.rsplit("|", 2)
    if len(parts) == 3:
        name, etype, box_field = parts
        return (name.strip() or None), etype.strip(), box_field.strip()
    if len(parts) == 2:
        name, etype = parts
        return (name.strip() or None), etype.strip(), None
    # 只有一段：当成长度不完整的实体名，类型未知
    name = frag.strip("[]").strip()
    return (name or None), "UNK", None


def _parse_json_prediction(text: str) -> Tuple[List[EntityTriple], List[str]]:
    """兼容 `{"entities": [{"name": ..., "type": ..., "box": [...]}]}` 这类输出。

    这里假设 JSON 里的框和 pipe 格式一样，是同一套（缩放后）像素坐标；
    如果模型按 Qwen 的 0~1000 归一化坐标输出，需要调用方先换算再喂给指标。
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return [], [text]
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], [text]

    items = obj.get("entities", obj) if isinstance(obj, dict) else obj
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return [], [text]

    entities: List[EntityTriple] = []
    malformed: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            malformed.append(str(item))
            continue
        name = item.get("name") or item.get("entity") or item.get("text") or item.get("span")
        etype = item.get("type") or item.get("label") or item.get("entity_type") or "UNK"
        if not name:
            malformed.append(json.dumps(item, ensure_ascii=False))
            continue
        regions = None
        for key in ("box", "boxes", "bbox", "region", "regions", "position"):
            if item.get(key) is not None:
                regions = _coerce_regions(item[key])
                break
        entities.append(EntityTriple(str(name), str(etype), regions))
    return entities, malformed


def _coerce_regions(value: Any) -> Optional[List[Box]]:
    if value is None:
        return None
    if isinstance(value, str):
        if is_none_field(value):
            return None
        try:
            return parse_regions(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return [tuple(int(round(float(v))) for v in value)]
        boxes: List[Box] = []
        for sub in value:
            box = _coerce_regions(sub)
            if box:
                boxes.extend(box)
        return boxes or None
    return None


# --------------------------------------------------------------------------- #
# 框与格式化
# --------------------------------------------------------------------------- #
def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_box_iou(pred_boxes: Sequence[Box], gold_boxes: Sequence[Box]) -> float:
    return max((box_iou(p, g) for p in pred_boxes for g in gold_boxes), default=0.0)


def rescale_entities(
    entities: Iterable[EntityTriple],
    scale_w: float,
    scale_h: float,
    width: int,
    height: int,
) -> List[EntityTriple]:
    """把预测框从"模型看到的坐标"换算回目标坐标系（通常是原图）。"""
    out: List[EntityTriple] = []
    for e in entities:
        if not e.regions:
            out.append(e)
            continue
        boxes = [scale_box(b, scale_w, scale_h, width, height) for b in e.regions]
        out.append(EntityTriple(e.text, e.etype, boxes))
    return out


def format_entities(entities: Sequence[EntityTriple]) -> str:
    """统一输出成标注格式 `entity|type|box;entity|type|None`，空结果输出 None。"""
    if not entities:
        return "None"
    return ";".join(f"{e.text}|{e.etype}|{format_regions(e.regions)}" for e in entities)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _prf(tp: int, n_pred: int, n_gold: int, prefix: str = "") -> Dict[str, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}f1": f1,
    }


def _sample_prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, float]:
    """单样本自己的 P/R/F1；两边都空（金标 None、预测也 None）算满分。"""
    if n_pred == 0 and n_gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return _prf(tp, n_pred, n_gold)


def _match(
    preds: Sequence[EntityTriple],
    golds: Sequence[EntityTriple],
    iou_threshold: Optional[float],
) -> Tuple[int, int, int, int, List[Tuple[str, str]]]:
    """一对一匹配，返回 (tp, fp, fn, box_errors, 命中项的 (name, type) 列表)。

    匹配用位掩码 DP 求最优解：先最大化匹配数，再最大化总分数（分数就是 IoU），
    避免逐个金标挑局部最优时"前面抢走了后面唯一能用的预测"这类次优结果。
    状态数是 O(num_pred × 2^num_gold)，只和单个样本的实体数有关。

    iou_threshold=None 表示只看文本（名字+类型）；否则金标带框时，预测也必须带框
    且 IoU 达标才算命中，此时用 IoU 当分数做并列时的取舍。
    """
    num_preds, num_golds = len(preds), len(golds)
    if num_preds == 0 or num_golds == 0:
        return 0, num_preds, num_golds, 0, []

    edges = _build_edges(preds, golds, iou_threshold)
    if not edges:
        # 一条可匹配的边都没有，但"名字类型对得上却因为框没对上"的情况仍要计入
        return 0, num_preds, num_golds, _box_error_count(preds, golds, set()), []

    @lru_cache(maxsize=None)
    def dp(i: int, used_mask: int) -> Tuple[int, float, int]:
        """处理 pred i..end，返回 (匹配数, 总分数, pred i 选中的 gold 下标或 -1)。"""
        if i >= num_preds:
            return (0, 0.0, -1)
        best = dp(i + 1, used_mask)  # 放弃 pred i
        for j in range(num_golds):
            if ((used_mask >> j) & 1) != 0:
                continue
            score = edges.get((i, j))
            if score is None:
                continue
            sub_cnt, sub_score, _ = dp(i + 1, used_mask | (1 << j))
            cand = (sub_cnt + 1, sub_score + float(score), j)
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
        return best

    # 回溯出具体匹配了哪些 (pred, gold)
    matched_pairs: List[Tuple[int, int]] = []
    used_mask, i = 0, 0
    while i < num_preds:
        _, _, j = dp(i, used_mask)
        if j >= 0:
            matched_pairs.append((i, j))
            used_mask |= 1 << j
        i += 1

    tp = len(matched_pairs)
    matched_golds = {j for _, j in matched_pairs}
    box_errors = _box_error_count(preds, golds, matched_golds)
    matched_keys = [entity_key(golds[j]) for _, j in matched_pairs]
    return tp, num_preds - tp, num_golds - tp, box_errors, matched_keys


def _box_error_count(
    preds: Sequence[EntityTriple],
    golds: Sequence[EntityTriple],
    matched_golds: Set[int],
) -> int:
    """诊断用：名字+类型能对上，但最后没配上的金标（框不达标 / 框缺失 / 被别的金标占用）。"""
    return sum(
        1
        for j, gold in enumerate(golds)
        if j not in matched_golds
        and any(entity_key(pred) == entity_key(gold) for pred in preds)
    )


def _build_edges(
    preds: Sequence[EntityTriple],
    golds: Sequence[EntityTriple],
    iou_threshold: Optional[float],
) -> Dict[Tuple[int, int], float]:
    """给"名字+类型一致"的 (pred i, gold j) 打分；不该匹配的组合不建边。"""
    edges: Dict[Tuple[int, int], float] = {}
    for i, pred in enumerate(preds):
        for j, gold in enumerate(golds):
            if entity_key(pred) != entity_key(gold):
                continue
            if iou_threshold is None:
                # 只看文本：只要名字+类型对上就能匹配，带框时用 IoU 做并列取舍
                edges[(i, j)] = 1.0 + (
                    best_box_iou(pred.regions, gold.regions)
                    if pred.regions and gold.regions
                    else 0.0
                )
                continue
            if gold.regions is None and pred.regions is None:
                edges[(i, j)] = 1.0  # 金标不可见，预测也必须输出 None
            elif gold.regions is not None and pred.regions is not None:
                iou = best_box_iou(pred.regions, gold.regions)
                if iou >= iou_threshold:
                    edges[(i, j)] = iou
    return edges


class GroundedMNERMetric:
    """累加式指标：每 update() 一个样本就当场算出该样本的指标并刷新累计指标。

    - update() 返回的 per-sample 记录里带这个样本自己的 precision/recall/f1，
      同时把累计（micro）指标刷新到 self.running；
    - compute() 只是读取累计结果，不再重新统计（多卡/离线复算走 update_record()，
      同样每加一条就刷新一次）。
    """

    def __init__(self, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.tp_text = self.fp_text = self.fn_text = 0
        self.tp_grounded = self.fp_grounded = self.fn_grounded = 0
        self.num_gold = self.num_pred = 0
        self.num_grounded_gold = self.num_grounded_pred = 0
        self.malformed = 0
        self.box_errors = 0
        self.per_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"gold": 0, "pred": 0, "tp": 0}
        )
        self.records: List[Dict[str, Any]] = []
        self.running: Dict[str, Any] = self._build_metrics()

    @property
    def running_metrics(self) -> Dict[str, Any]:
        """当前累计指标（每加一个样本就更新过，随时可读）。"""
        return self.running

    # -- 累加 ------------------------------------------------------------- #
    def update(
        self,
        gold_text: Optional[str],
        pred_entities: Sequence[EntityTriple],
        pred_raw: str = "",
        malformed: Iterable[str] = (),
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """比较一条样本：gold_text 是原始标注，pred_entities 已换算到同一坐标系。"""
        golds, gold_malformed = parse_annotations(gold_text)
        preds = list(pred_entities)

        tp, fp, fn, box_errors, matched_keys = _match(preds, golds, self.iou_threshold)
        tp_text, fp_text, fn_text, _, matched_text_keys = _match(preds, golds, None)
        grounded_preds = [p for p in preds if p.regions]
        grounded_golds = [g for g in golds if g.regions]
        tp_g, fp_g, fn_g, _, _ = _match(grounded_preds, grounded_golds, self.iou_threshold)

        record: Dict[str, Any] = {
            "gold": gold_text,
            "pred": format_entities(preds),
            "pred_raw": pred_raw,
            # 这个样本自己的指标，加进来的时候就算好
            **_sample_prf(tp, len(preds), len(golds)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tp_text": tp_text,
            "fp_text": fp_text,
            "fn_text": fn_text,
            "tp_grounded": tp_g,
            "fp_grounded": fp_g,
            "fn_grounded": fn_g,
            "num_gold": len(golds),
            "num_pred": len(preds),
            "num_grounded_gold": len(grounded_golds),
            "num_grounded_pred": len(grounded_preds),
            "malformed": len(list(malformed)) + len(gold_malformed),
            "box_errors": box_errors,
            "tp_types": [etype for _, etype in matched_keys],
        }
        if meta:
            record.update(meta)
        self.update_record(record)
        return record

    def update_record(self, record: Dict[str, Any]) -> None:
        """把一条 per-sample 记录累加进统计量（也用于汇总多卡结果 / 离线打分）。"""
        for key in (
            "tp",
            "fp",
            "fn",
            "tp_text",
            "fp_text",
            "fn_text",
            "tp_grounded",
            "fp_grounded",
            "fn_grounded",
            "num_gold",
            "num_pred",
            "num_grounded_gold",
            "num_grounded_pred",
            "malformed",
            "box_errors",
        ):
            setattr(self, key, getattr(self, key) + record.get(key, 0))
        for text, key in ((record.get("gold"), "gold"), (record.get("pred"), "pred")):
            entities, _ = parse_annotations(text)
            for entity in entities:
                self.per_type[normalize_type(entity.etype)][key] += 1
        for etype in record.get("tp_types") or []:
            self.per_type[normalize_type(etype)]["tp"] += 1
        self.records.append(record)
        # 累计指标也在这里刷新，后面 compute() 只读不算
        self.running = self._build_metrics()

    # -- 汇总 ------------------------------------------------------------- #
    def _build_metrics(self) -> Dict[str, Any]:
        """按当前累计量算一份指标快照；每次 update 一个样本都会调用一次。"""
        metrics: Dict[str, Any] = {}
        metrics.update(_prf(self.tp, self.num_pred, self.num_gold))
        metrics.update(_prf(self.tp_text, self.num_pred, self.num_gold, "text_"))
        metrics.update(
            _prf(self.tp_grounded, self.num_grounded_pred, self.num_grounded_gold, "grounded_")
        )
        metrics.update(
            {
                "num_gold": self.num_gold,
                "num_pred": self.num_pred,
                "num_grounded_gold": self.num_grounded_gold,
                "malformed": self.malformed,
                "box_errors": self.box_errors,
            }
        )
        metrics["per_type"] = {}
        for etype, v in sorted(self.per_type.items()):
            entry = _prf(v["tp"], v["pred"], v["gold"])
            entry.update({"gold": v["gold"], "pred": v["pred"], "tp": v["tp"]})
            metrics["per_type"][etype] = entry
        return metrics

    def compute(self, prefix: str = "") -> Dict[str, Any]:
        """读取累计指标：数值在 update 时就算好了，这里只按 prefix 改 key 名。"""
        if not prefix:
            return dict(self.running)
        return {f"{prefix}{key}": value for key, value in self.running.items()}

    def summary(self, prefix: str = "") -> str:
        metrics = self.compute(prefix)
        return (
            f"{prefix}f1={metrics[prefix + 'f1']:.4f} "
            f"(P={metrics[prefix + 'precision']:.4f} R={metrics[prefix + 'recall']:.4f}) | "
            f"text_f1={metrics[prefix + 'text_f1']:.4f} | "
            f"grounded_f1={metrics[prefix + 'grounded_f1']:.4f} | "
            f"gold={metrics[prefix + 'num_gold']} pred={metrics[prefix + 'num_pred']} "
            f"malformed={metrics[prefix + 'malformed']} box_err={metrics[prefix + 'box_errors']}"
        )


# --------------------------------------------------------------------------- #
# 落盘 / 离线打分
# --------------------------------------------------------------------------- #
def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_records(
    records: Sequence[Dict[str, Any]], iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> Tuple[GroundedMNERMetric, Dict[str, Any]]:
    """对已保存的预测记录重新打分：用记录里的 gold 与 pred 文本重算指标。

    匹配用规范化后的 pred（坐标已经换算回原图），格式错误数 malformed 则看
    pred_raw（模型原始输出），这样离线复算的结果和当时在线跑的完全一致。
    """
    metric = GroundedMNERMetric(iou_threshold=iou_threshold)
    for record in records:
        pred_text = record.get("pred") or record.get("pred_raw") or ""
        pred_entities, _ = parse_prediction(pred_text)
        _, malformed = parse_prediction(record.get("pred_raw") or pred_text)
        metric.update(
            gold_text=record.get("gold"),
            pred_entities=pred_entities,
            pred_raw=record.get("pred_raw", ""),
            malformed=malformed,
            meta={k: record[k] for k in ("id", "image", "index") if record.get(k) is not None},
        )
    return metric, metric.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="对保存的预测结果离线打分")
    parser.add_argument("--pred", required=True, help="predictions_*.jsonl 预测文件")
    parser.add_argument("--gold", default=None, help="可选：金标 jsonl，按行覆盖预测里的 gold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="框匹配 IoU 阈值")
    parser.add_argument("--prefix", default="test", help="指标前缀")
    args = parser.parse_args()

    records = load_records(args.pred)
    if args.gold:
        with open(args.gold, "r", encoding="utf-8") as f:
            golds = [json.loads(line)["messages"][1]["content"] for line in f if line.strip()]
        for record, gold in zip(records, golds):
            record["gold"] = gold

    metric, metrics = score_records(records, iou_threshold=args.iou)
    print(metric.summary(args.prefix))
    print(json.dumps(metrics[args.prefix + "per_type"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
