from .parser import FuncDefValidator
from .trie import Trie, TrieNode
from typing import List, Tuple, Dict, Union
from llm_sdk import Small_LLM_Model
from .masking import chose_next_token, logits_masking
from .valid_number import valid_next_ids_for_number
from .valid_string import valid_next_ids_for_string


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


def generate_func_name(llm: Small_LLM_Model,
                       trie: Trie,
                       prompt_ids: List[int]) \
        -> Tuple[List[int], Union[FuncDefValidator, str]]:

    curr_ids: List[int] = prompt_ids.copy()
    generated_ids: List[int] = []
    curr_node: TrieNode = trie.root

    while not curr_node.is_end_leaf:
        logits = llm.get_logits_from_input_ids(curr_ids)
        valid_ids = trie.valid_next_ids(curr_node)
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
                    stop_token_ids: List[int]) -> Tuple[List[int], float]:

    curr_state: str = "START"
    generated_ids: List[int] = []
    generated_str: str = ""

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

        curr_state = candidate_ids[next_token]
        generated_ids.append(next_token)
        curr_ids.append(next_token)
        generated_str += valid_vocab[next_token]

    return generated_ids, float(generated_str)


def generate_string(llm: Small_LLM_Model,
                    curr_ids: List[int],
                    vocab: Dict[str, int],
                    quote_token_ids: int) -> Tuple[List[int], str]:

    curr_state: str = "IN_STRING"
    generated_ids: List[int] = []

    while True:
        logits = llm.get_logits_from_input_ids(curr_ids)
        candidate_ids = valid_next_ids_for_string(vocab, curr_state)

        allowed_ids = set(candidate_ids.keys())
        if curr_state == "IN_STRING":
            allowed_ids.add(quote_token_ids)

        masked = logits_masking(logits, allowed_ids)
        next_token = chose_next_token(masked)

        if next_token == quote_token_ids:
            curr_ids.append(next_token)
            break

        curr_state = candidate_ids[next_token]
        generated_ids.append(next_token)
        curr_ids.append(next_token)
        if len(generated_ids) >= 3 and generated_ids[-1] == generated_ids[-2] == generated_ids[-3]:
            break

        print(f"state={curr_state}, len={len(curr_ids)}, token={llm.decode([next_token])!r}")

    return generated_ids, llm.decode(generated_ids)


def generate_boolean(llm: Small_LLM_Model,
                     curr_ids: List[int],
                     bool_trie: Trie) -> Tuple[List[int], str]:

    curr_node: TrieNode = bool_trie.root
    generated_ids: List[int] = []

    while not curr_node.is_end_leaf:
        logits = llm.get_logits_from_input_ids(curr_ids)
        valid_ids = bool_trie.valid_next_ids(curr_node)
        masked = logits_masking(logits, valid_ids)
        next_id = chose_next_token(masked)
        generated_ids.append(next_id)
        curr_ids.append(next_id)
        curr_node = bool_trie.advance(next_id, curr_node)

    assert isinstance(curr_node.func_def, str)
    return generated_ids, curr_node.func_def
