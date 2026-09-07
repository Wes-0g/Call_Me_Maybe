from argparse import ArgumentParser, Namespace, Action
from sys import exit
from llm_sdk import Small_LLM_Model
from typing import Callable, List
from .parser import (JsonParser,
                     FuncDefValidator,
                     FuncCallValidator,
                     ParserError)
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
    print(f"\n{'='*92}\n")

    result = constrained_decoding(llm, func_def_list, func_call_list)



if __name__ == "__main__":
    main()


'''
uv run python -m src [--functions_definition <function_definition_file>]
                     [--input <input_file>] [--output <output_file>]
'''
