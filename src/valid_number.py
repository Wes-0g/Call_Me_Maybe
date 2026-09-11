from typing import Dict, Optional


def number_state(state: str, char: str) -> str | None:
    if state in ("START", "AFTER_MINUS"):
        if char == '-' and state == 'START':
            return "AFTER_MINUS"
        if char == '0':
            return "INT_ZERO"
        if char in '123456789':
            return "INT_NONZERO"
    elif state == "INT_ZERO":
        if char == '.':
            return "AFTER_DOT"
    elif state == "INT_NONZERO":
        if char in '0123456789':
            return "INT_NONZERO"
        if char == '.':
            return "AFTER_DOT"
    elif state in ("AFTER_DOT", "FRAC_DIGIT"):
        if char in '0123456789':
            return "FRAC_DIGIT"

    return None


def is_valid_number(state: str, token_str: str) -> Optional[str]:
    curr: Optional[str] = state

    for char in token_str:
        assert curr is not None
        curr = number_state(curr, char)
        if curr is None:
            return None

    return curr


def build_number_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    allowed_chars: str = "-0123456789."
    valid: Dict[int, str] = {}

    for token_str, token_id in vocab.items():
        if not token_str.strip():
            continue
        if all(char in allowed_chars for char in token_str):
            valid[token_id] = token_str

    return valid


def valid_next_ids_for_number(valid_vocab: Dict[int, str],
                              curr_state: str,
                              integer_only: bool = False
                              ) -> Dict[int, str]:

    valid: Dict[int, str] = {}

    for token_id, token_str in valid_vocab.items():
        if integer_only and "." in token_str:
            continue
        state = is_valid_number(curr_state, token_str)
        if state is not None:
            valid[token_id] = state

    return valid
