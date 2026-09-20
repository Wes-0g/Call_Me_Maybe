"""Load and validate function definitions, requests, and vocabulary."""

from json import load, JSONDecodeError
from argparse import Namespace
from typing import List, Dict, Tuple, Any, cast
from enum import Enum
from sys import exit
try:
    from llm_sdk import Small_LLM_Model
    from pydantic import (BaseModel,
                          model_validator,
                          ValidationError,
                          ConfigDict)
except ModuleNotFoundError:
    print("llm_sdk or pydantic is not installed, install them first to run the program.")
    exit(1)


class ParserError(Exception):
    """Represent an input parsing or validation failure.

    Args:
        message: Original parsing or validation error information.

    Returns:
        A ParserError carrying the supplied message.

    Attributes:
        message: Original parsing or validation information.
    """

    def __init__(self, message: Any) -> None:
        """Store the original message and initialize the exception.

        Args:
            message: Original error information to store and pass to Exception.

        Returns:
            None.
        """
        self.message: Any = message
        super().__init__(self.message)


class ParamType(Enum):
    """List the parameter and return types accepted by function schemas.

    Args:
        value: Supported type string: number, string, boolean, or integer.

    Returns:
        Matching ParamType enumeration member.

    Raises:
        ValueError: If value is not a supported type string.

    Attributes:
        NUMBER: Numeric type with value "number".
        STRING: String type with value "string".
        BOOLEAN: Boolean type with value "boolean".
        integer: Integer type with value "integer".
    """

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    integer = "integer"


class FuncDefParameter(BaseModel):
    """Validate a parameter type and reject unknown fields.

    Args:
        type: Supported parameter type, as a ParamType or accepted string.

    Returns:
        Validated parameter model.

    Raises:
        ValidationError: If the type is missing, unsupported, or extra fields
            are supplied.

    Attributes:
        type: Validated parameter type.
    """

    model_config = ConfigDict(extra='forbid')
    type: ParamType


class FuncDefReturn(BaseModel):
    """Validate a return type and reject unknown fields.

    Args:
        type: Supported return type, as a ParamType or accepted string.

    Returns:
        Validated return-type model.

    Raises:
        ValidationError: If the type is missing, unsupported, or extra fields
            are supplied.

    Attributes:
        type: Validated return type.
    """

    model_config = ConfigDict(extra='forbid')
    type: ParamType


class FuncDefValidator(BaseModel):
    """Validate a function name, description, parameters, and return type.

    Args:
        name: Nonblank function name that does not start with an ASCII digit.
        description: Nonblank description of what the function does.
        parameters: Mapping of parameter names to parameter models or
            dictionaries.
        returns: Return-type model or a dictionary describing the return type.

    Returns:
        Validated function-definition model.

    Raises:
        ValidationError: If fields, names, or the function description fail
            validation.

    Attributes:
        name: Function name.
        description: Function description.
        parameters: Validated parameter definitions.
        returns: Validated return definition.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, FuncDefParameter]
    returns: FuncDefReturn

    @model_validator(mode='after')
    def data_validator(self) -> "FuncDefValidator":
        """Check function metadata and return the validated model.

        Raise ValueError for blank names or descriptions, or for function and
        parameter names that is not an identifer.

        Args:
            None.

        Returns:
            This function-definition model after metadata validation.

        Raises:
            ValueError: If metadata is blank or a function/parameter name
                is not an identifer.
        """
        if not self.name.strip():
            raise ValueError("Function name cannot be empty.")
        if not self.name.isidentifier():
            raise ValueError("Function name must be an identifier.")
        if not self.description.strip():
            raise ValueError("Function description cannot be empty.")
        for param_name in self.parameters.keys():
            if not param_name.strip():
                raise ValueError("Parameter name cannot be empty.")
            if not param_name.isidentifier():
                raise ValueError("Parameter name name must be an identifier.")
        return self


class FuncCallValidator(BaseModel):
    """Validate a request containing a nonblank prompt and no extra fields.

    Args:
        prompt: Nonblank natural-language request.

    Returns:
        Validated request model.

    Raises:
        ValidationError: If prompt is missing or blank, or extra fields are
            supplied.

    Attributes:
        prompt: Validated request text.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str

    @model_validator(mode='after')
    def data_validator(self) -> "FuncCallValidator":
        """Validate a nonblank prompt or raise ValueError.

        Args:
            None.

        Returns:
            This request model after prompt validation.

        Raises:
            ValueError: If prompt contains only whitespace or is empty.
        """
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return self


class JsonParser:
    """Load function definitions and requests from CLI-supplied file paths.

    Args:
        args: CLI namespace containing functions_definition and input paths.

    Returns:
        A parser configured with the supplied input paths.

    Attributes:
        args: CLI namespace with the configured input paths.
    """

    def __init__(self, args: Namespace) -> None:
        """Store the parsed CLI arguments containing the input file paths.

        Args:
            args: CLI namespace containing functions_definition and input
                paths.

        Returns:
            None.
        """
        self.args: Namespace = args

    @staticmethod
    def no_duplicate(data: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Build a dictionary from JSON key-value pairs.

        Raise ParserError when a key occurs more than once in the same object.

        Args:
            data: Ordered key-value pairs from one decoded JSON object.

        Returns:
            Dictionary containing each key once.

        Raises:
            ParserError: If a key appears more than once.
        """
        seen: set[str] = set()
        build_dict: Dict[str, Any] = {}

        for key, value in data:
            if key in seen:
                raise ParserError(f"Duplicate key: {key} -> {value}")
            seen.add(key)
            build_dict[key] = value
        return build_dict

    def func_def_parser(self) -> List[FuncDefValidator]:
        """Load and validate the configured function definitions.

        Reject an empty input and duplicate object keys. Raise ParserError for
        handled file access, JSON decoding, or model validation failures.

        Args:
            None.

        Returns:
            List of validated function-definition models.

        Raises:
            ParserError: If a handled file, JSON, duplicate-key, empty-input,
                or model validation error occurs.
        """
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

        seen: set = set()
        for func in func_def_json:
            name = func.get('name')
            if name in seen:
                raise ParserError(f"Duplicated Function name '{name}'")
            seen.add(name)

        try:
            return [FuncDefValidator(**item) for item in func_def_json]
        except ValidationError as e:
            raise ParserError(e)
        except TypeError:
            raise ParserError("Function definitions data must be "
                              "a list of dictionaries.")

    def func_call_parser(self) -> List[FuncCallValidator]:
        """Load and validate the configured natural-language requests.

        Reject an empty input and duplicate object keys. Raise ParserError for
        handled file access, JSON decoding, or model validation failures.

        Args:
            None.

        Returns:
            List of validated request models.

        Raises:
            ParserError: If a handled file, JSON, duplicate-key, empty-input,
                or model validation error occurs.
        """
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
        except TypeError:
            raise ParserError("Function calls data must be "
                              "a list of dictionaries.")


def load_vocab(llm: Small_LLM_Model) -> Dict[str, int]:
    """Load the SDK vocabulary as a mapping from token text to token IDs.

    Exit if the vocabulary is missing, unreadable, or invalid JSON. The type
    cast documents the expected shape without validating it at runtime.

    Args:
        llm: SDK model providing the vocabulary file path.

    Returns:
        Mapping from vocabulary token text to token IDs.

    Raises:
        SystemExit: If the file is missing, unreadable, or contains invalid
            JSON.
    """
    try:
        with open(llm.get_path_to_vocab_file()) as file:
            return cast(Dict[str, int], load(file))
    except FileNotFoundError:
        exit("Vocabulary file not found.")
    except PermissionError:
        exit("Vocabulary file not readable.")
    except JSONDecodeError:
        exit("Vocabulary file is not a valid JSON file.")
