"""Build prompts and generate constrained function names and values."""

from .parser import FuncDefValidator
from .trie import Trie, TrieNode
from typing import List, Tuple, Dict, Union
from llm_sdk import Small_LLM_Model
from .masking import chose_next_token, logits_masking
from .valid_number import valid_next_ids_for_number


def prompt_builder(func_defs: List[FuncDefValidator], prompt: str) -> str:
    """Build a function-selection prompt with signatures and input guidance.

    Args:
        func_defs: Available validated function definitions.
        prompt: Natural-language request to include in the prompt.

    Returns:
        Prompt text ending with the function-name JSON prefix.
    """
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


def generate_func_name_or_bool(llm: Small_LLM_Model,
                               trie: Trie,
                               prompt_ids: List[int]) \
        -> Tuple[List[int], Union[FuncDefValidator, str]]:
    """Return generated token IDs and the value selected from the trie.

    Work on a copy of prompt_ids, leaving the supplied sequence unchanged.
    Use model scores at branches and append forced tokens directly.
    Raise ValueError if a transition or the selected value is missing.

    Args:
        llm: SDK model used to score ambiguous token choices.
        trie: Trie containing allowed function names or boolean literals.
        prompt_ids: Complete input context; this list is not modified.

    Returns:
        Pair of generated token IDs and the selected definition or string.

    Raises:
        ValueError: If a trie transition or terminal value is missing.
    """
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
                    ) -> Union[int, float]:
    """Generate a number and append its tokens to curr_ids.

    Return an int when integer_only is true, otherwise return a float.
    Append a selected stop token to the context before returning. At the
    token limit, convert the collected text without appending a stop token;
    incomplete numeric text can raise ValueError during conversion.

    Args:
        llm: SDK model used to obtain next-token scores.
        curr_ids: Complete context, extended in place as tokens are generated.
        valid_vocab: Mapping of numeric token IDs to their text.
        stop_token_ids: Token IDs allowed to terminate a complete number.
        max_tokens: Maximum generation steps. Defaults to 64.
        integer_only: Whether to forbid decimals and return an int. Defaults to
            False.

    Returns:
        Generated integer or floating-point value.

    Raises:
        ValueError: If the collected text cannot be converted to the requested
            numeric type.
    """
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
            return value

        curr_state = candidate_ids[next_token]
        generated_ids.append(next_token)
        curr_ids.append(next_token)
        generated_str += valid_vocab[next_token]

    return int(generated_str) if integer_only else float(generated_str)


def generate_string(llm: Small_LLM_Model,
                    curr_ids: List[int],
                    content_ids: set[int],
                    quote_ids: set[int],
                    closing_quote: int,
                    max_tokens: int = 128) -> str:
    """Generate string content until a quote-bearing token or the token limit.

    Expect the opening quote to be present in curr_ids. Preserve content
    before the first quote in the final token and append closing_quote.
    At the token limit, return collected text without appending a quote.
    Treat escaped quotes as terminators and leave escape sequences literal.

    Args:
        llm: SDK model used to score and decode tokens.
        curr_ids: Context ending in an opening quote, extended in place.
        content_ids: Allowed token IDs without double quotes.
        quote_ids: Token IDs containing a terminating double quote.
        closing_quote: Standalone quote token appended on normal termination.
        max_tokens: Maximum generation steps. Defaults to 128.

    Returns:
        Decoded string content, including text before the terminating quote.
    """
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

    value = llm.decode(generated_ids)
    return value


def generate_boolean(llm: Small_LLM_Model,
                     curr_ids: List[int],
                     bool_trie: Trie) -> str:
    """Return the boolean text selected by bool_trie and extend curr_ids.

    Reuse function-name trie traversal and append its generated IDs to the
    original context. Raise ValueError if the selected value is not a string.

    Args:
        llm: SDK model used for trie selection.
        curr_ids: Complete context, extended with the selected literal.
        bool_trie: Trie containing the encoded true and false literals.

    Returns:
        Selected boolean text, either "true" or "false".

    Raises:
        ValueError: If trie selection fails or the selected value is not a
            string.
    """
    generated_ids, value = generate_func_name_or_bool(llm, bool_trie, curr_ids)
    if not isinstance(value, str):
        raise ValueError("Boolean trie did not return a boolean string")

    curr_ids.extend(generated_ids)
    return value
