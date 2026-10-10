from dataclasses import dataclass
from typing import List, Optional, Tuple


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[List[Box]]
    region_valid: bool = True

