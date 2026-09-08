from llm_sdk import Small_LLM_Model
from typing import Dict, Tuple, Union, Any, List
from .parser import (FuncDefValidator,
                     FuncDefParameter,
                     ParamType,
                     FuncCallValidator,
                     load_vocab)
from .generate import (prompt_builder,
                       generate_string,
                       generate_number,
                       generate_boolean,
                       generate_func_name)
from .valid_boolean import build_boolean_trie
from .valid_number import build_number_vocab
from .trie import Trie


def build_output_entry(prompt: str,
                       func_def: FuncDefValidator,
                       params: Dict[str, Union[float, str, bool]]
                       ) -> Dict[str, Any]:

    return {
        "prompt": prompt,
        "name": func_def.name,
        "parameters": params
    }


def generate_params(llm: Small_LLM_Model,
                    curr_ids: List[int],
                    func_def: FuncDefValidator,
                    number_vocab: Dict[int, str],
                    vocab: Dict[str, int],
                    bool_trie: Trie,
                    quote_id: int,
                    comma_ids: List[int],
                    closing_ids: List[int]
                    ) -> Dict[str, Any]:

    params_value: Dict[str, Union[float, str, bool]] = {}
    params_items: List[
        Tuple[str, FuncDefParameter]
    ] = list(func_def.parameters.items())

    for i, (param_name, param) in enumerate(params_items):
        prefix = f'"{param_name}": '
        if param.type == ParamType.STRING:
            prefix += '"'
        prefix_ids = llm.encode(prefix)[0].tolist()
        curr_ids.extend(prefix_ids)

        if param.type == ParamType.STRING:
            _, value = generate_string(llm, curr_ids, vocab, quote_id, 30)
        elif param.type == ParamType.NUMBER:
            _, value = generate_number(
                llm, curr_ids, number_vocab, comma_ids + closing_ids)
        elif param.type == ParamType.BOOLEAN:
            _, value = generate_boolean(llm, curr_ids, bool_trie)
        else:
            raise ValueError(f"Invalid parameter type: {param.type}")
        params_value[param_name] = value

        # if i < len(params_items) - 1:
        # curr_ids.extend(comma_ids)

    return params_value


def process_prompt(llm: Small_LLM_Model,
                   func_call: FuncCallValidator,
                   func_defs: List[FuncDefValidator],
                   number_vocab: Dict[int, str],
                   vocab: Dict[str, int],
                   name_trie: Trie,
                   bool_trie: Trie,
                   quote_id: int,
                   comma_ids: List[int],
                   closing_ids: List[int]
                   ) -> Dict[str, Any]:

    prompt_text: str = prompt_builder(func_defs, func_call.prompt)
    prompt_ids: List[int] = llm.encode(prompt_text)[0].tolist()

    name_ids, func_def = generate_func_name(llm, name_trie, prompt_ids)

    curr_ids: List[int] = prompt_ids + name_ids

    param_value: Dict[
        str, Any
    ] = generate_params(llm, curr_ids,
                        func_def, number_vocab,
                        vocab, bool_trie, quote_id,
                        comma_ids, closing_ids)

    return build_output_entry(func_call.prompt, func_def, param_value)


def constrained_decoding(llm: Small_LLM_Model,
                         func_defs: List[FuncDefValidator],
                         func_calls: List[FuncCallValidator]
                         ) -> List[Dict[str, Any]]:

    name_trie: Trie = Trie()
    name_trie.trie_builder(func_defs, llm)
    bool_trie: Trie = build_boolean_trie(llm)
    vocab: Dict[str, int] = load_vocab(llm)
    number_vocab: Dict[int, str] = build_number_vocab(vocab)
    quote_id: int = llm.encode('"')[0].tolist()[0]
    comma_ids: List[int] = llm.encode(',')[0].tolist()
    closing_ids: List[int] = llm.encode('}}')[0].tolist()

    result: List[Dict[str, Any]] = []
    for func_call in func_calls:
        entry = process_prompt(llm, func_call,
                               func_defs, number_vocab,
                               vocab, name_trie,
                               bool_trie, quote_id,
                               comma_ids, closing_ids)
        print(entry)
        result.append(entry)
    return result
