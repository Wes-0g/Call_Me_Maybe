"""Validate numeric token continuations with a finite-state machine."""

from typing import Dict, Optional


def number_state(state: str, char: str) -> str | None:
    """Advance one numeric character or return None if it is illegal.

    Args:
        state: Current numeric FSM state name.
        char: Single character to consume.

    Returns:
        Next state name, or None if the transition is invalid.
    """
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
    """Walk token_str from state and return the resulting state or None.

    A returned state may describe an unfinished number, such as a lone minus
    sign; the generation loop decides whether the number can terminate.

    Args:
        state: Numeric FSM state before reading the token.
        token_str: Token text to validate character by character.

    Returns:
        Resulting state, possibly incomplete, or None for invalid text.
    """
    curr: Optional[str] = state

    for char in token_str:
        assert curr is not None
        curr = number_state(curr, char)
        if curr is None:
            return None

    return curr


def build_number_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Return nonblank tokens containing only numeric characters.

    Args:
        vocab: Mapping of vocabulary token text to token IDs.

    Returns:
        Mapping of numeric token IDs to nonblank text containing only minus,
            digits, and decimal points.
    """
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
    """Map legal numeric token IDs to their resulting FSM states.

    Exclude tokens containing a decimal point when integer_only is true.

    Args:
        valid_vocab: Mapping of numeric token IDs to their text.
        curr_state: Current numeric FSM state.
        integer_only: Whether to exclude decimal-point tokens. Defaults to
            False.

    Returns:
        Mapping of legal next-token IDs to their resulting states.
    """
    valid: Dict[int, str] = {}

    for token_id, token_str in valid_vocab.items():
        if integer_only and "." in token_str:
            continue
        state = is_valid_number(curr_state, token_str)
        if state is not None:
            valid[token_id] = state

    return valid
