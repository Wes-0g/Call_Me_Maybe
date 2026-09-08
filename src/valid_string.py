from typing import Dict, Optional, Union


def build_string_vocab(vocab: Dict[str, int]
                       ) -> Dict[str, Dict[int, Union[str, None]]]:
    valid_vocab: Dict[str, Dict[int, Union[str, None]]] = {}

    for state in ['IN_STRING', "AFTER_BACKSLASH"]:
        valid_vocab[state] = {
            token_id: is_valid_string(state, token_str)
            for token_str, token_id in vocab.items()
            if is_valid_string(state, token_str) is not None
        }

    return valid_vocab


def string_state(state: str, char: str) -> Optional[str]:
    if state == "IN_STRING":
        if char == '\\':
            return "AFTER_BACKSLASH"
        if char == '"':
            return None
        return "IN_STRING"
    if state == "AFTER_BACKSLASH":
        if char in '"\\/bfnrt':
            return "IN_STRING"
        return None
    return None


def is_valid_string(state: str, token_str: str) -> Optional[str]:

    curr: Optional[str] = state
    for char in token_str:
        assert curr is not None
        curr = string_state(curr, char)
        if curr is None:
            return None

    return curr


def valid_next_ids_for_string(vocab: Dict[str, int],
                              curr_state: str
                              ) -> Dict[int, Union[str, None]]:

    valid: Dict[
        int, Union[str, None]
    ] = build_string_vocab(vocab)[curr_state]

    return valid
