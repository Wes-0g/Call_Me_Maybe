from json import load, dump
from argparse import ArgumentParser, Namespace
from typing import Callable
try:
    from pydantic import Field, model_validator, field_validator, json, json_schema
except ImportError:
    print("pydantic is not installed. Please install it using pip.")
    exit(1)


class Parser:
    pass


def main() -> None:
    arg_parser: ArgumentParser = ArgumentParser(description="")
    add_argument: Callable = arg_parser.add_argument
    add_argument("--functions_definition",
                 default="data/input/functions_definition.json",
                 help="Path to the functions definition JSON file.", )
    add_argument("--input",
                 default="data/input/function_calling_tests.json",
                 help="Path to the input JSON file.")
    add_argument("--output",
                 default="data/output/function_calling_results.json",
                 help="Path to the output JSON file.")
    x = arg_parser.parse_args()
    print(x.functions_definition)
    print(x.input)
    print(x.output)


if __name__ == "__main__":
    main()



'''
uv run python -m src [--functions_definition <function_definition_file>]
                     [--input <input_file>] [--output <output_file>]
'''
