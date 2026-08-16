from json import load, JSONDecodeError
from argparse import Namespace
from typing import List, Dict, Any
from enum import Enum
try:
    from pydantic import (BaseModel,
                          model_validator,
                          ValidationError,
                          ConfigDict)
except ImportError:
    print("pydantic is not installed. Please install it using pip.")
    exit(1)


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
        if not self.name.startswith('fn_'):
            raise ValueError("Function name must start with 'fn_'.")
        if not self.description.strip():
            raise ValueError("Function description cannot be empty.")
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

    def func_def_parser(self) -> List[FuncDefValidator]:
        try:
            with open(self.args.functions_definition, 'r') as file:
                func_def_json: List[Dict[str, Any]] = load(file)
        except JSONDecodeError as e:
            raise ValueError()
        try:
            return [FuncDefValidator(**item) for item in func_def_json]
        except ValidationError as e:
            raise ValueError()

    def func_call_parser(self) -> List[FuncCallValidator]:
        try:
            with open(self.args.input, 'r') as file:
                func_call_json: List[Dict[str, str]] = load(file)
        except JSONDecodeError as e:
            raise ValueError()
        try:
            return [FuncCallValidator(**item) for item in func_call_json]
        except ValidationError as e:
            raise ValueError()
