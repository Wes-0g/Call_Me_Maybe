from argparse import ArgumentParser, Namespace, Action
from typing import Callable, List
from .parser import (JsonParser,
                     FuncDefValidator,
                     FuncCallValidator,
                     ParserError)


def main() -> None:
    arg_parser: ArgumentParser = ArgumentParser(description="")
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
        print(e)
        exit(1)

    # print(func_def_list)
    # print()
    # print(func_call_list)


if __name__ == "__main__":
    main()


'''
uv run python -m src [--functions_definition <function_definition_file>]
                     [--input <input_file>] [--output <output_file>]
'''
