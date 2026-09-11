from typing import List, Set
from math import inf
import numpy as np


def logits_masking(logits: List[float], valid_token_ids: Set[int]) \
        -> List[float]:
    return [
        logit if i in valid_token_ids else -inf
        for i, logit in enumerate(logits)
    ]


def chose_next_token(masked_logits: List[float]) -> int:
    return int(np.argmax(masked_logits))
