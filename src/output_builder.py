"""Assemble function calls using reusable model and token data."""

from typing import Any, Dict, List, Union
from llm_sdk import Small_LLM_Model
from .parser import (FuncCallValidator,
                     FuncDefValidator,
                     ParamType,
                     load_vocab)
from .generate import (prompt_builder,
                       generate_func_name_or_bool,
                       generate_string,
                       generate_number,
                       generate_boolean)
from .trie import Trie
from .valid_boolean import build_boolean_trie
from .valid_number import build_number_vocab
from .valid_string import split_string_tokens


class FunctionCaller:
    """Group the model, function schemas, and reusable decoding data.

    Args:
        llm: SDK model shared across requests.
        functions: Validated function definitions available for selection.

    Returns:
        A FunctionCaller with initialized tries and token filters.

    Attributes:
        llm: Shared SDK model.
        functions: Available function definitions.
        names: Function-name trie.
        booleans: Boolean-literal trie.
        numbers: Numeric token ID-to-text mapping.
        string_ids: Content-token IDs without quotes.
        quote_ids: Token IDs containing quotes.
        quote: Standalone quote token ID.
        comma: Encoded comma separator.
        closing: Encoded closing brace.
    """

    def __init__(self, llm: Small_LLM_Model,
                 functions: List[FuncDefValidator]) -> None:
        """Store the model and functions, then prepare tries and filters.

        Args:
            llm: SDK model shared across requests.
            functions: Validated function definitions used to build the name
                trie.

        Returns:
            None.

        Raises:
            SystemExit: If the SDK vocabulary cannot be loaded by load_vocab.
        """
        self.llm: Small_LLM_Model = llm
        self.functions: List[FuncDefValidator] = functions
        self.names: Trie = Trie()
        self.names.trie_builder(functions, llm)
        self.booleans: Trie = build_boolean_trie(llm)
        vocab: Dict[str, int] = load_vocab(llm)
        self.numbers: Dict[int, str] = build_number_vocab(vocab)
        self.string_ids, self.quote_ids = split_string_tokens(vocab)
        self.quote: int = llm.encode('"')[0].tolist()[0]
        self.comma: List[int] = llm.encode(',')[0].tolist()
        self.closing: List[int] = llm.encode('}')[0].tolist()

    def generate_params(self, ids: List[int],
                        function: FuncDefValidator) -> Dict[str, Any]:
        """Generate typed arguments and extend ids in place.

        Inject parameter names and separators from the schema. Strip string
        whitespace and convert boolean text to bool.
        Raise ValueError for an unsupported parameter type.

        Args:
            ids: Model context, extended in place with keys, values, and
                separators.
            function: Selected function defining parameter names and types.

        Returns:
            Mapping of parameter names to typed values, with strings stripped.

        Raises:
            ValueError: If a parameter type is unsupported or numeric/trie
                decoding fails.
        """
        arguments: Dict[str, Any] = {}
        params = list(function.parameters.items())
        value: Union[str, int, float, bool]

        for index, (name, param) in enumerate(params):
            separator: List[int] = self.comma if (
                    index < len(params) - 1) else self.closing
            is_number: bool = param.type in (
                ParamType.NUMBER, ParamType.integer)
            prefix: str = f'"{name}": '
            if param.type == ParamType.STRING:
                prefix += '"'
            ids.extend(self.llm.encode(prefix)[0].tolist())

            if param.type == ParamType.STRING:
                value = generate_string(
                    self.llm, ids, self.string_ids, self.quote_ids, self.quote)
            elif is_number:
                value = generate_number(
                    self.llm, ids, self.numbers, separator,
                    integer_only=param.type == ParamType.integer)
            elif param.type == ParamType.BOOLEAN:
                text = generate_boolean(self.llm, ids, self.booleans)
                value = text == 'true'
            else:
                raise ValueError(f'Invalid parameter type: {param.type}')
            if param.type == ParamType.STRING:
                assert isinstance(value, str)
                value = value.strip()
            arguments[name] = value
            if not is_number:
                ids.extend(separator)

        return arguments

    def process_prompt(self, request: FuncCallValidator) -> Dict[str, Any]:
        """Return the selected function and arguments for one request.

        Build the model context and select a function before generating its
        parameters. Raise ValueError if selection does not return a definition.

        Args:
            request: Validated natural-language request to process.

        Returns:
            Dictionary containing prompt, name, and parameters.

        Raises:
            ValueError: If function selection or argument decoding fails.
        """
        prompt: str = prompt_builder(self.functions, request.prompt)
        ids: List[int] = self.llm.encode(prompt)[0].tolist()

        name_ids, function = generate_func_name_or_bool(
            self.llm, self.names, ids)
        if not isinstance(function, FuncDefValidator):
            raise ValueError('Decoding did not select a function definition')

        ids.extend(name_ids)
        ids.extend(self.llm.encode('\", "parameters": {')[0].tolist())
        return {
            'prompt': request.prompt,
            'name': function.name,
            'parameters': self.generate_params(ids, function),
        }


def constrained_decoding(llm: Small_LLM_Model,
                         func_defs: List[FuncDefValidator],
                         func_calls: List[FuncCallValidator]
                         ) -> List[Dict[str, Any]]:
    """Generate and return output entries for all requests in input order.

    Reuse one FunctionCaller for the batch and print each completed entry.
    Let parsing, model, and decoding errors propagate to the caller.

    Args:
        llm: SDK model shared by the batch.
        func_defs: Available validated function definitions.
        func_calls: Validated requests to process in input order.

    Returns:
        List of generated output dictionaries in request order.

    Raises:
        SystemExit: If vocabulary loading fails during caller setup.
        ValueError: If function selection or argument decoding fails.
    """
    caller: FunctionCaller = FunctionCaller(llm, func_defs)
    results: List[Dict[str, Any]] = []
    for request in func_calls:
        entry = caller.process_prompt(request)
        print(entry)
        results.append(entry)

    return results
