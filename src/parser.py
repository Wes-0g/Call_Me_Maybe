from json import load, JSONDecodeError
from argparse import Namespace
from typing import List, Dict, Tuple, Any
from enum import Enum
try:
    from pydantic import (BaseModel,
                          model_validator,
                          ValidationError,
                          ConfigDict)
except ImportError:
    print("pydantic is not installed. Please install it using pip.")
    exit(1)


class ParserError(Exception):
    def __init__(self, message: Any) -> None:
        self.message: Any = message
        super().__init__(self.message)


class ParamType(Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class FuncDefParameter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: ParamType


class FuncDefReturn(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: ParamType


class FuncDefValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, FuncDefParameter]
    returns: FuncDefReturn

    @model_validator(mode='after')
    def data_validator(self) -> "FuncDefValidator":
        if not self.name.strip():
            raise ValueError("Function name cannot be empty.")
        if self.name.startswith(tuple([str(x) for x in range(10)])):
            raise ValueError("Function name cannot start with a number.")
        if not self.description.strip():
            raise ValueError("Function description cannot be empty.")
        for param_name in self.parameters.keys():
            if not param_name.strip():
                raise ValueError("Parameter name cannot be empty.")
            if param_name.startswith(tuple([str(x) for x in range(10)])):
                raise ValueError("Parameter name cannot start with a number.")
        return self


class FuncCallValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @model_validator(mode='after')
    def data_validator(self) -> "FuncCallValidator":
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return self


class JsonParser:

    def __init__(self, args: Namespace) -> None:
        self.args: Namespace = args

    @staticmethod
    def no_duplicate(data: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        seen: set[str] = set()
        build_dict: Dict[str, Any] = {}

        for key, value in data:
            if key in seen:
                raise ParserError(f"Duplicate key: {key} -> {value}")
            seen.add(key)
            build_dict[key] = value
        return build_dict

    def func_def_parser(self) -> List[FuncDefValidator]:
        try:
            with open(self.args.functions_definition, 'r') as file:
                func_def_json: List[
                    Dict[str, Any]
                ] = load(file, object_pairs_hook=self.no_duplicate)
                if not func_def_json:
                    raise ParserError("No function definitions found.")
        except FileNotFoundError:
            raise ParserError("Function definitions file not found.")
        except PermissionError:
            raise ParserError("Permission denied to read function "
                              "definitions file.")
        except IsADirectoryError:
            raise ParserError("Function definitions file is a directory.")
        except JSONDecodeError:
            raise ParserError("Invalid JSON format.")
        try:
            return [FuncDefValidator(**item) for item in func_def_json]
        except ValidationError as e:
            raise ParserError(e)

    def func_call_parser(self) -> List[FuncCallValidator]:
        try:
            with open(self.args.input, 'r') as file:
                func_call_json: List[
                    Dict[str, str]
                ] = load(file, object_pairs_hook=self.no_duplicate)
                if not func_call_json:
                    raise ParserError("No function calls found.")
        except FileNotFoundError:
            raise ParserError("Function calls file not found.")
        except PermissionError:
            raise ParserError("Permission denied to read function "
                              "calls file.")
        except IsADirectoryError:
            raise ParserError("Function calls file is a directory.")
        except JSONDecodeError:
            raise ParserError("Invalid JSON format.")
        try:
            return [FuncCallValidator(**item) for item in func_call_json]
        except ValidationError as e:
            raise ParserError(e)
