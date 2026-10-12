from typing import Any, Dict, List, Optional
DEFAULT_SYSTEM_PROMPT = (
    "You are an information extraction assistant for grounded "
    "multimodal named entity recognition. "
    "Given the image and text, first think briefly and then answer "
    "in a strict XML-like format. "
    "The final output must contain exactly one "
    "<think>...</think> block followed by exactly one "
    "<answer>...</answer> block. "
    "Inside <answer>, output only structured entity triples in the format "
    "mention|type|[x1,y1,x2,y2], separated by '; '. "
    "If there is no entity, output <answer>None</answer>. "
    "Do not add any extra explanation outside the tags."
)
