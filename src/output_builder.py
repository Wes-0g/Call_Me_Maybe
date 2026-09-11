from typing import Any, Dict, List
from llm_sdk import Small_LLM_Model
from .parser import FuncCallValidator, FuncDefValidator, ParamType, load_vocab
from .generate import (prompt_builder, generate_func_name, generate_string,
                       generate_number, generate_boolean)
from .trie import Trie
from .valid_boolean import build_boolean_trie
from .valid_number import build_number_vocab
from .valid_string import split_string_tokens


class FunctionCaller:
    """Keep the model and reusable token pools together."""

    def __init__(self, llm: Small_LLM_Model,
                 functions: list[FuncDefValidator]) -> None:
        self.llm = llm
        self.functions = functions
        self.names = Trie()
        self.names.trie_builder(functions, llm)
        self.booleans = build_boolean_trie(llm)
        vocab = load_vocab(llm)
        self.numbers = build_number_vocab(vocab)
        self.string_ids, self.quote_ids = split_string_tokens(vocab)
        self.quote = llm.encode('"')[0].tolist()[0]
        self.comma = llm.encode(',')[0].tolist()
        self.closing = llm.encode('}')[0].tolist()

    def generate_params(self, ids: list[int],
                        function: FuncDefValidator) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        params = list(function.parameters.items())

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
                value: str | int | float | bool = generate_string(
                    self.llm, ids, self.string_ids, self.quote_ids, self.quote)
            elif is_number:
                _, value = generate_number(
                    self.llm, ids, self.numbers, separator,
                    integer_only=param.type == ParamType.integer)
            elif param.type == ParamType.BOOLEAN:
                _, text = generate_boolean(self.llm, ids, self.booleans)
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
        prompt: str = prompt_builder(self.functions, request.prompt)
        ids: List[int] = self.llm.encode(prompt)[0].tolist()

        name_ids, function = generate_func_name(self.llm, self.names, ids)
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

    caller: FunctionCaller = FunctionCaller(llm, func_defs)
    results: List[Dict[str, Any]] = []
    for request in func_calls:
        entry = caller.process_prompt(request)
        print(entry)
        results.append(entry)

    return results
