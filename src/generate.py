from valid_number import valid_next_ids_for_number
from .parser import FuncDefValidator
from .trie import Trie, TrieNode
from typing import List, Tuple, Dict
from llm_sdk import Small_LLM_Model
from .masking import chose_next_token, logits_masking


def generate_func_name(llm: Small_LLM_Model,
                       trie: Trie,
                       prompt_ids: List[int]) \
        -> Tuple[List[int], FuncDefValidator]:

    curr_ids: List[int] = prompt_ids.copy()
    generated_ids: List[int] = []
    curr_node: TrieNode = trie.root

    while not curr_node.is_end_leaf:
        logits = llm.get_logits_from_input_ids(curr_ids)
        valid_ids = trie.valid_next_ids_for_name(curr_node)
        masked = logits_masking(logits, valid_ids)
        next_token = chose_next_token(masked)

        generated_ids.append(next_token)
        curr_ids.append(next_token)
        curr_node = trie.advance(next_token, curr_node)

    if curr_node.func_def is None:
        raise ValueError("Function definition not found")
    return generated_ids, curr_node.func_def


def generate_number(llm: Small_LLM_Model,
                    curr_ids: List[int],
                    valid_vocab: Dict[int, str],
                    stop_token_ids: List[int]) -> List[int]:

    curr_state: str = "START"
    generated_ids: List[int] = []

    while True:
        logits = llm.get_logits_from_input_ids(curr_ids)
        candidate_ids = valid_next_ids_for_number(valid_vocab, curr_state)

        allowed_ids = set(candidate_ids.keys())
        if curr_state in ["INT_ZERO", "INT_NONZERO", "FRAC_DIGIT"]:
            for stop_id in stop_token_ids:
                allowed_ids.add(stop_id)

        masked = logits_masking(logits, allowed_ids)
        next_token = chose_next_token(masked)

        if next_token in stop_token_ids:
            break

        new_state = candidate_ids[next_token]
        generated_ids.append(next_token)
        curr_ids.append(next_token)
        curr_state = new_state

    return generated_ids
