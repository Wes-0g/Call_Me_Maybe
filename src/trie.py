"""Store allowed token sequences for constrained selection."""

from typing import Optional, List, Dict, Set, Union
from llm_sdk import Small_LLM_Model
from .parser import FuncDefValidator


class TrieNode:
    """Hold child token edges and an optional value for a complete sequence.

    Args:
        None.

    Returns:
        An empty nonterminal TrieNode.

    Attributes:
        children: Mapping of token IDs to child nodes.
        is_end_leaf: Whether the node ends an inserted sequence.
        func_def: Associated definition or literal, or None.
    """

    def __init__(self) -> None:
        """Create a node with no children or terminal value.

        Args:
            None.

        Returns:
            None.
        """
        self.children: Dict[int, TrieNode] = {}
        self.is_end_leaf = False
        self.func_def: Optional[Union[FuncDefValidator, str]] = None


class Trie:
    """Store allowed token sequences and their associated values.

    Args:
        None.

    Returns:
        A Trie containing an empty root node.

    Attributes:
        root: Root node for all inserted token sequences.
    """

    def __init__(self) -> None:
        """Create an empty root node.

        Args:
            None.

        Returns:
            None.
        """
        self.root = TrieNode()

    def insert(self, ids: list[int],
               func_def: Union[FuncDefValidator, str]) -> None:
        """Insert a token sequence and attach func_def to its terminal node.

        Args:
            ids: Token sequence to insert from the root.
            func_def: Definition or literal attached to the terminal node.

        Returns:
            None. Update the trie in place.
        """
        curr = self.root

        for id_ in ids:
            if id_ not in curr.children:
                curr.children[id_] = TrieNode()
            curr = curr.children[id_]
        curr.is_end_leaf = True
        curr.func_def = func_def

    def trie_builder(self, func_defs: List[FuncDefValidator],
                     llm: Small_LLM_Model) -> None:
        """Encode each function name and insert it with its definition.

        Args:
            func_defs: Function definitions whose names should be inserted.
            llm: SDK model used to encode function names.

        Returns:
            None. Add the encoded names to this trie.
        """
        for func_def in func_defs:
            ids = llm.encode(func_def.name)[0].tolist()
            self.insert(ids, func_def)

    @staticmethod
    def valid_next_ids(curr_node: TrieNode) -> Set[int]:
        """Return the token IDs available from curr_node.

        Args:
            curr_node: Node whose outgoing token edges should be inspected.

        Returns:
            Set of token IDs leading to child nodes.
        """
        return set(curr_node.children)

    @staticmethod
    def advance(token_id: int, curr_node: TrieNode) -> Optional[TrieNode]:
        """Return the child for token_id, or None if that edge is absent.

        Args:
            token_id: ID of the outgoing edge to follow.
            curr_node: Node from which to follow the edge.

        Returns:
            Matching child node, or None if the edge does not exist.
        """
        return curr_node.children.get(token_id)
