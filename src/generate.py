from .parser import FuncDefValidator
from .trie import Trie, TrieNode
from typing import List, Tuple, Dict, Union
from llm_sdk import Small_LLM_Model
from .masking import chose_next_token, logits_masking
from .valid_number import valid_next_ids_for_number


def prompt_builder(func_defs: List[FuncDefValidator], prompt: str) -> str:
    feeding_prompt = (
        "Choose a function and extract its input arguments as JSON. "
        "Do not calculate the result. Preserve the requested input text. "
        "For regex replacement, regex is the pattern to match "
        "(use [0-9]+ for numbers); replacement is the inserted text.\n"
        "Available functions:\n"
    )
    for func in func_defs:
        parameters = ", ".join(
            f"{name}: {param.type.value}"
            for name, param in func.parameters.items()
        )
        feeding_prompt += f"{func.name}({parameters}): {func.description}\n"
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
        valid_ids = trie.valid_next_ids(curr_node)
        if len(valid_ids) == 1:
            next_token = list(valid_ids)[0]
        else:
            logits = llm.get_logits_from_input_ids(curr_ids)
            next_token = chose_next_token(logits_masking(logits, valid_ids))

        generated_ids.append(next_token)
        curr_ids.append(next_token)
        following = trie.advance(next_token, curr_node)
        if following is None:
            raise ValueError("Invalid function trie transition")
        curr_node = following

    if curr_node.func_def is None:
        raise ValueError("Function definition not found")
    return generated_ids, curr_node.func_def


def generate_number(llm: Small_LLM_Model,
                    curr_ids: List[int],
                    valid_vocab: Dict[int, str],
                    stop_token_ids: List[int],
                    max_tokens: int = 64,
                    integer_only: bool = False
                    ) -> Tuple[List[int], Union[int, float]]:

    curr_state: str = "START"
    generated_ids: List[int] = []
    generated_str: str = ""

    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(curr_ids)
        candidate_ids = valid_next_ids_for_number(
            valid_vocab, curr_state, integer_only)

        allowed_ids = set(candidate_ids.keys())
        if curr_state in ["INT_ZERO", "INT_NONZERO", "FRAC_DIGIT"]:
            allowed_ids.update(stop_token_ids)

        masked = logits_masking(logits, allowed_ids)
        next_token = chose_next_token(masked)

        if next_token in stop_token_ids:
            curr_ids.append(next_token)
            value = (int(generated_str) if integer_only
                     else float(generated_str))
            return generated_ids, value

        curr_state = candidate_ids[next_token]
        generated_ids.append(next_token)
        curr_ids.append(next_token)
        generated_str += valid_vocab[next_token]

    return generated_ids, value


def generate_string(llm: Small_LLM_Model,
                    curr_ids: List[int],
                    content_ids: set[int],
                    quote_ids: set[int],
                    closing_quote: int,
                    max_tokens: int = 128) -> str:
    """Generate content until the model selects a token containing a quote."""
    generated_ids: List[int] = []
    allowed_ids: set[int] = content_ids.union(quote_ids)
    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(curr_ids)
        token = chose_next_token(logits_masking(logits, allowed_ids))
        if token in quote_ids:
            ending: str = llm.decode([token]).split('"', 1)[0]
            curr_ids.extend(llm.encode(ending)[0].tolist())
            curr_ids.append(closing_quote)
            value: str = llm.decode(generated_ids)
            return value + ending
        curr_ids.append(token)
        generated_ids.append(token)

    return value + ending


def generate_boolean(llm: Small_LLM_Model,
                     curr_ids: List[int],
                     bool_trie: Trie) -> Tuple[List[int], str]:

    generated_ids, value = generate_func_name(llm, bool_trie, curr_ids)
    if not isinstance(value, str):
        raise ValueError("Boolean trie did not return a boolean string")

    curr_ids.extend(generated_ids)
    return generated_ids, value
