from json import load, dump, JSONDecodeError
from argparse import ArgumentParser, Namespace
from typing import Callable, List, Dict, Union
try:
    from pydantic import BaseModel, Field, model_validator, field_validator
except ImportError:
    print("pydantic is not installed. Please install it using pip.")
    exit(1)


class FuncDefParameter(BaseModel):
    pass


class FuncDefValidator(BaseModel):
    name: str
    description: str


class FuncCallValidator(BaseModel):
    prompt: str


class JsonParser:

    def __init__(self, args: Namespace) -> None:
        self.args: Namespace = args

    def parse(self) -> None:
        self.func_def_parser()
        self.func_call_parser()

    def func_def_parser(self) -> None:
        with open(self.args.functions_definition, 'r') as file:
            func_def_json: List[
                Dict[str, Union[str, Dict[str, str]]]
            ] = load(file)

    def func_call_parser(self) -> None:
        with open(self.args.input, 'r') as file:
            func_call_json: List[Dict[str, str]] = load(file)


def main() -> None:
    arg_parser: ArgumentParser = ArgumentParser(description="")
    add_argument: Callable = arg_parser.add_argument
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
    json_parser.parse()


if __name__ == "__main__":
    main()


'''
uv run python -m src [--functions_definition <function_definition_file>]
                     [--input <input_file>] [--output <output_file>]
'''
