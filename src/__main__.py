import json
from argparse import ArgumentParser, Action, Namespace
from typing import Callable, List, Dict, Any
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .parser import (JsonParser,
                     ParserError,
                     FuncDefValidator,
                     FuncCallValidator)
from .output_builder import constrained_decoding


def main() -> None:
    arg_parser: ArgumentParser = ArgumentParser(
        description="Function calling assistant."
    )
    add_argument: Callable[..., Action] = arg_parser.add_argument
    add_argument("--functions_definition",
                 default="data/input/functions_definition.json",
                 help="Path to the functions definition JSON file.")
    add_argument("--input",
                 default="data/input/function_calling_tests.json",
                 help="Path to the input JSON file.")
    add_argument("--output",
                 default="data/output/function_calling_results.json",
                 help="Path to the output JSON file.")

    args: Namespace = arg_parser.parse_args()
    json_parser: JsonParser = JsonParser(args)
    try:
        func_def_list: List[
            FuncDefValidator] = json_parser.func_def_parser()
        func_call_list: List[
            FuncCallValidator] = json_parser.func_call_parser()
    except ParserError as e:
        exit(f"{e}")

    llm: Small_LLM_Model = Small_LLM_Model()

    results: List[
        Dict[str, Any]
    ] = constrained_decoding(llm, func_def_list, func_call_list)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4), encoding='utf-8')


if __name__ == '__main__':
    main()
