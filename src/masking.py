"""Mask illegal token scores and select the highest-scoring token."""

from typing import List, Set
from math import inf
import numpy as np


def logits_masking(logits: List[float], valid_token_ids: Set[int]) \
        -> List[float]:
    """Return scores with negative infinity at IDs outside valid_token_ids.

    Args:
        logits: Next-token scores indexed by token ID.
        valid_token_ids: IDs whose original scores should remain selectable.

    Returns:
        New score list with negative infinity at disallowed IDs.
    """
    return [
        logit if i in valid_token_ids else -inf
        for i, logit in enumerate(logits)
    ]


def chose_next_token(masked_logits: List[float]) -> int:
    """Return the index of the highest score as a Python int.

    Use the first index when scores tie. NumPy raises ValueError for an
    empty input; an all-negative-infinity input still selects index zero.

    Args:
        masked_logits: Token scores after masking.

    Returns:
        Index of the first highest-scoring token as a Python int.

    Raises:
        ValueError: If masked_logits is empty.
    """
    return int(np.argmax(masked_logits))
