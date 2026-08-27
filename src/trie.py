from typing import Optional, List, Dict
from llm_sdk import Small_LLM_Model
from .parser import FuncDefValidator

class TrieNode:

    def __init__(self) -> None:
        self.children: Dict[int, TrieNode] = {}
        self.is_end_leaf = False
        self.func_def: Optional[FuncDefValidator] = None


class Trie:

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, ids: list[int], func_def: FuncDefValidator) -> None:
        curr = self.root

        for id_ in ids:
            if id_ not in curr.children:
                curr.children[id_] = TrieNode()
            curr = curr.children[id_]
        curr.is_end_leaf = True
        curr.func_def = func_def

    def trie_builder(self, func_defs: List[FuncDefValidator], llm: Small_LLM_Model) -> None:

        for func_def in func_defs:
            ids = llm.encode(func_def.name)[0].tolist()
            self.insert(ids, func_def)

    def some_shit(self) -> None:
        pass