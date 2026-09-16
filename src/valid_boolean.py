"""Build the token trie for JSON boolean values."""

from llm_sdk import Small_LLM_Model
from .trie import Trie


def build_boolean_trie(llm: Small_LLM_Model) -> Trie:
    """Return a trie containing the encoded true and false literals.

    Args:
        llm: SDK model used to encode the boolean literals.

    Returns:
        Trie containing true and false token sequences and their text values.
    """
    trie: Trie = Trie()

    trie.insert(llm.encode('true')[0].tolist(), 'true')
    trie.insert(llm.encode('false')[0].tolist(), 'false')

    return trie
