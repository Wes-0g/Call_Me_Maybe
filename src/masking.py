from typing import List, Set
from math import inf
from .parser import FuncDefValidator


def logits_masking(logits: List[float], valid_token_ids: Set[int]) \
        -> List[float]:
    return [
        logit if i in valid_token_ids else -inf
        for i, logit in enumerate(logits)
    ]


def chose_next_token(m_logits: List[float]) -> int:
    # return m_logits.index(max(m_logits))
    return max(enumerate(m_logits), key=lambda pair: pair[1])[0]


def prompt_builder(func_defs: List[FuncDefValidator], prompt: str) -> str:
    feeding_prompt: str = ("You are a function-calling assistant. "
                           "Choose the correct function for "
                           "the user's request.\n"
                           "Available functions:\n")
    feeding_prompt += "\n".join(
        [f"- {func.name}: {func.description}" for func in func_defs]
    )
    feeding_prompt += f"\nUser request: {prompt}\n"
    feeding_prompt += '{"function_name": "'
    return feeding_prompt
