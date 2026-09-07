from llm_sdk import Small_LLM_Model
from .trie import Trie


def build_boolean_trie(llm: Small_LLM_Model) -> Trie:
    trie: Trie = Trie()

    trie.insert(llm.encode('true')[0].tolist(), 'true')
    trie.insert(llm.encode('false')[0].tolist(), 'false')

    return trie
