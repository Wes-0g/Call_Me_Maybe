"""Run function-call generation from the command line."""

import json
from argparse import ArgumentParser, Namespace
from typing import List, Dict, Any
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .parser import (JsonParser,
                     ParserError,
                     FuncDefValidator,
                     FuncCallValidator)
from .output_builder import constrained_decoding
from sys import exit
import time


def main() -> None:
    """Parse CLI options, generate function calls, and write the output JSON.

    Create the output directory when needed and print the elapsed time.
    Exit on an input parsing error; other failures propagate to the caller.

    Args:
        None.

    Returns:
        None. Write the output file and print elapsed time.

    Raises:
        SystemExit: If parsed inputs are invalid.
        OSError: If the output directory or file cannot be written.
    """
    start: float = time.time()

    arg_parser: ArgumentParser = ArgumentParser(
        description="Function calling assistant."
    )
    arg_parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.")
    arg_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file.")
    arg_parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.")
    arg_parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model identifier"
    )

    args: Namespace = arg_parser.parse_args()
    json_parser: JsonParser = JsonParser(args)
    try:
        func_def_list: List[
            FuncDefValidator] = json_parser.func_def_parser()
        func_call_list: List[
            FuncCallValidator] = json_parser.func_call_parser()
    except ParserError as e:
        exit(f"{e}")

    llm: Small_LLM_Model = Small_LLM_Model(args.model)

    results: List[
        Dict[str, Any]
    ] = constrained_decoding(llm, func_def_list, func_call_list)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4))

    print(f"\ntime: {time.time() - start:.2f}s")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(" KeyboardInterrupt ...")
    except Exception as e:
        exit(f"ERROR: {type(e).__name__}: {e}")
