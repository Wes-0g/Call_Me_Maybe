*This project has been created as part of the 42 curriculum by zel-fati.*

# CALL ME MAYBE

## Description

**Call Me Maybe** converts natural-language requests into structured function
calls using a local language model and constrained decoding.

### Goal

Given a list of function definitions and user requests, select the appropriate
function and extract its typed input arguments. For example, “What is the sum
of 2 and 3?” becomes a call to `fn_add_numbers` with `a = 2` and `b = 3`.

The program generates a description of the call; it does not execute the
selected function. The supplied moulinette executes supported calls to compare
results with its expected answers.

### Overview

The project includes:

- JSON input parsing and Pydantic validation, including duplicate-key detection.
- A token trie for selecting function names and boolean literals.
- A finite-state machine for generating numbers and integers.
- A simple string decoder using content-token and quote-token sets.
- Logit masking and greedy token selection.
- A CLI that writes results to JSON and reports elapsed time.

Inference uses `Qwen/Qwen3-0.6B` through the bundled `llm_sdk` package. The
application accesses model encoding, decoding, and scores through this SDK;
`src/` does not import PyTorch or Transformers directly. See the
[Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B) for model information.

## Instructions

### Requirements

- Python **3.10 or later**.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency
  installation and execution.
- `make` for the convenience commands below.
- The bundled `llm_sdk/` directory, declared as a local uv workspace member.
- Network access for dependency installation and the initial model download.

There is no separate application compilation step. Model weights are cached
locally after downloading. The SDK selects an available MPS or CUDA device,
otherwise it uses the CPU; execution time depends on the machine.

### Installation

From the repository root:

```bash
make install
```

Or run the underlying command:

```bash
uv sync
```

### Execution

```bash
make run
```

Equivalent command:

```bash
uv run python -m src
```

Default paths:

| Option | Default |
| --- | --- |
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

The output directory is created automatically. Results are printed while
processing, and the completed list is written after generation finishes.

```bash
uv run python -m src --help
```

### Code Quality

```bash
make lint
```

This runs flake8 and mypy against `src/` using the project environment.

### Debug Mode

Use the module entry point directly:

```bash
uv run python -m pdb -m src
```

The existing `make debug` recipe still points to `main.py`; use the command
above for the current package layout.

### Clean

```bash
make clean
```

This removes Python bytecode caches and mypy caches.

## Example Usage

### Input and Output

A function definition file contains a JSON array of schemas:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "returns": {"type": "number"}
    }
]
```

A request file contains a JSON array of prompts:

```json
[
    {"prompt": "What is the sum of 2 and 3?"}
]
```

Example output:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    }
]
```

Supported parameter types are `number`, `integer`, `string`, and `boolean`.

### Custom Files

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/public_results.json
```

### Private Input Set

```bash
uv run python -m src \
  --functions_definition moulinette/data/input/functions_definition.json \
  --input moulinette/data/input/function_calling_tests.json \
  --output data/output/private_results.json
```

Using different output paths prevents the private grader from accidentally
reading answers produced for the public prompts.

## Algorithm Explanation

### 1. Validate Inputs and Prepare Reusable Data

`JsonParser` reads the two input files. An `object_pairs_hook` detects duplicate
keys before they can be silently overwritten, and Pydantic models check the
required fields, supported types, and nonblank metadata.

`FunctionCaller` then prepares the name trie, boolean trie, numeric vocabulary,
string-token sets, and structural quote/comma/brace IDs once for the batch.

### 2. Build the Model Context

The feeding prompt contains short extraction instructions, function signatures,
descriptions, and the user's request. It tells the model to supply input values
rather than calculate the requested result.

The prompt ends with:

```text
{"function_name": "
```

The next tokens therefore continue the selected function name. The final saved
key is `name`; this is assembled separately in Python.

### 3. Mask Illegal Tokens

At each model-driven choice, the decoder obtains a score, or **logit**, for each
token. It preserves scores for allowed IDs and replaces all other scores with
negative infinity:

```text
masked_score[token] = score[token]   if token is allowed
masked_score[token] = -infinity     otherwise
next_token = argmax(masked_score)
```

The implementation uses `numpy.argmax` for greedy selection. There is no random
sampling. Masking limits the possible continuations; the model's scores decide
which legal continuation to take.

Every SDK scoring call receives the full accumulated context: the original
prompt, injected structure, and tokens generated so far.

### 4. Select a Function or Boolean Through a Trie

Each permitted name is encoded into token IDs and inserted into a trie. The
current node's children are the allowed next tokens. A single child is appended
directly; branches are resolved using masked model scores.

Traversal ends at a terminal node carrying the selected function definition.
The same traversal is reused for the encoded `true` and `false` literals. This
is token-prefix matching, not matching user-request keywords.

### 5. Generate Arguments According to Their Types

Parameter names and order come from the selected schema. The caller injects
keys and separators; only argument values are chosen by the model.

**Numbers and integers:** a finite-state machine tracks whether the decoder is
at the start, after a minus sign, inside an integer, after a decimal point, or
inside fractional digits. It walks every character of a candidate numeric token.
Only complete numeric states may accept a terminating comma or brace. Integer
mode rejects tokens containing a decimal point and converts the result directly
with `int()`, avoiding an intermediate floating-point conversion.

**Strings:** the existing vocabulary is split into two sets: tokens without a
quote and tokens containing a quote. The caller supplies the opening quote.
Content tokens extend the value until a quote-bearing token is selected. Text
before that token's first quote is retained, one closing quote is appended to
the context, and the caller adds the separator. This preserves content bundled
with the closing quote, such as the last parenthesis of a pattern. The caller
strips leading and trailing whitespace from returned string values.

**Booleans:** the boolean trie supplies literal text, which the caller converts
to a Python `bool`.

### 6. Assemble and Serialize the Result

The result is built as a Python dictionary containing `prompt`, `name`, and
`parameters`. The CLI serializes the list with `json.dumps(..., indent=4)`.
Serialization escapes Python string values; it does not correct a semantically
wrong argument. See the [Python JSON documentation](https://docs.python.org/3/library/json.html)
for encoding and decoding behavior.

## Design Decisions

- **Readable, separate components:** parsing, tries, numeric validation,
  masking, generation, and output assembly have distinct modules.
- **One reusable caller:** `FunctionCaller` owns the model and precomputed
  token data, avoiding long argument lists and repeated setup.
- **Pydantic at input boundaries:** schemas and requests are validated models;
  internal trie nodes and FSM states remain plain Python structures.
- **A simple string decoder:** two token sets replace a full JSON-string FSM.
  This reduces complexity but has the limitations listed below.
- **Plain-text prompting:** function signatures and brief instructions provide
  context without chat-template or thinking markers.
- **Exact integer conversion:** integer arguments remain integers, including
  values that a float could not represent exactly.
- **Greedy decoding:** model scores select among legal tokens, with no
  hand-written function-selection rules.
- **Bounded generation:** numbers have a default budget of 64 steps and strings
  128 steps. These are token limits, not character limits.

## Performance Analysis

### Accuracy

A fresh run on September 15, 2026 passed **11/11 public tests (100%)** with
the current implementation and default input files. Generation wrote to a
separate temporary file, which was then checked with the supplied public grader.

The local public/private sets are small samples, not guarantees of accuracy on
unseen requests. Function-name constraints prevent selecting an unlisted name,
but they cannot ensure that the selected function or arguments are correct.
Private grading was not rerun for this README revision.

### Speed

The same 11-request public run reported **93.55 seconds** of total elapsed
time in the local development environment, about **8.50 seconds per request**
when dividing the batch total. This average includes setup; it is not an
isolated inference-latency benchmark.

The CLI prints `time: ...s` for the complete run, including input loading,
model initialization, generation, and output writing. Initial downloads, device
selection, CPU threading, prompt length, and generated text length affect time;
a result from one machine is not a universal five-minute guarantee.

Setup scans the vocabulary once and builds tries from the encoded names. For
vocabulary size `V`, a masked selection creates an `O(V)` score list and performs
an `O(V)` argmax. Numeric candidate checking scans the smaller numeric pool.
Trie storage is proportional to the total number of inserted token edges.
Model forward passes over the accumulated context are the main practical cost.
Following forced trie edges avoids unnecessary model calls.

### Reliability and Current Limits

- JSON output is assembled from typed Python values, but intermediate string
  generation is not a complete JSON grammar validator.
- Escaped quotes can terminate strings early, and JSON escape text is not
  unescaped into characters by the string decoder.
- String values are stripped, which can remove intentional surrounding spaces.
- Scientific notation is not part of the numeric FSM.
- At the token limit, strings return collected text without a forced closing
  quote. Numbers convert collected text without forcing a separator; incomplete
  text can raise `ValueError`. Later context can therefore be incomplete.
- The current argmax helper does not reject an all-negative-infinity score list;
  such a list selects its first index.
- Output is written only after the batch finishes. A failed run can leave an
  older result file at the requested path.

## Challenges Faced

**Extracting inputs rather than answers:** the small model initially supplied
`4` for “square root of 16.” The prompt was changed to explicitly request input
arguments and prohibit calculating the result.

**Keeping model context consistent:** missing commas or the bridge between a
function name and its parameters damaged later predictions. The caller now
injects that bridge and normally records each completed value's separator.

**Tokens spanning punctuation:** a token can contain both the end of a value
and its closing quote. Discarding the whole token lost useful characters. The
string decoder now keeps text before the first quote.

**Integer schemas:** private inputs introduced `integer` parameters. Integer
mode was added to numeric masking and direct conversion.

**Excessive string-validation complexity:** earlier versions rebuilt or
maintained state-specific token pools. The current version uses two precomputed
sets, accepting reduced escape handling in exchange for simpler code.

**Code organization:** storing reusable data in `FunctionCaller` simplified the
pipeline. This structure and the two-set string approach were inspired by the
local `monarch_call_me` reference project supplied during development.

## Testing Strategy

### Static Checks

```bash
make lint
```

This checks style and source annotations. It passed during preparation of this
README on September 15, 2026.

### Unit and Regression Tests

```bash
uv run python -m unittest discover -s tests -v
```

The test files use a mock SDK with scripted logits to examine token selection,
context updates, numeric types, quote handling, and token limits. Parser tests
cover malformed inputs, duplicate keys, empty fields, and integer schemas.

**Current status:** the suite is not fully passing. The latest run executed 13
tests and reported 3 failures and 5 errors, including errors in subtests. Some
checks still expect the previous `(ids, value)` return format, stricter error
handling, special-token filtering, and exceptions at the token limit. They need
to be reconciled with the current implementation; earlier passing counts do not
apply to this revision.

### End-to-End Grading

First generate the separate public/private result files shown in Example Usage.
Then, from the repository root:

```bash
cd moulinette

uv run python -m moulinette grade_student_answers \
  --student_answer_path ../data/output/public_results.json

uv run python -m moulinette grade_student_answers -set private \
  --student_answer_path ../data/output/private_results.json
```

The grader checks prompt correspondence and evaluates the selected calls
against expected results. A valid JSON file can still receive a low score when
argument values are wrong. Always check that generation succeeded before grading.

## Project Structure

```text
.
├── src/
│   ├── __main__.py       # CLI, output writing, elapsed time
│   ├── parser.py         # Input models and JSON loading
│   ├── trie.py           # Token-prefix storage and traversal
│   ├── masking.py        # Illegal-score masking and argmax
│   ├── generate.py       # Feeding prompt and generation loops
│   ├── output_builder.py # FunctionCaller and batch orchestration
│   ├── valid_number.py   # Numeric FSM and candidate filtering
│   ├── valid_string.py   # Content/quote token sets
│   └── valid_boolean.py  # Boolean trie construction
├── data/input/           # Default schemas and requests
├── data/output/          # Generated results
├── llm_sdk/              # Local model SDK workspace package
├── moulinette/           # Supplied grader and additional inputs
├── tests/                # Parser and decoding regression tests
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Resources

### Documentation and References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original
  Transformer paper, for the model architecture behind token prediction.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — documentation
  for the model used by the SDK.
- [Python JSON documentation](https://docs.python.org/3/library/json.html) —
  serialization, decoding, and object-pair hooks.
- [Pydantic models](https://pydantic.dev/docs/validation/latest/concepts/models/)
  — typed input models and validation.
- [NumPy argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
  — selecting the highest-scoring token index.
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
  — installing the dependency and environment tool.

The organization of this README was inspired by the author's `fly-in` README:
project description, practical commands, algorithm details, project structure,
and explicit references and AI usage.

### AI Usage

AI assistance was used for both implementation and documentation:

- **Debugging and implementation:** diagnosing missing context delimiters,
  square-root argument errors, string-token boundaries, and integer support in
  `generate.py`, `output_builder.py`, and the validation helpers.
- **Refactoring:** simplifying the feeding prompt, string decoder, and caller
  structure, and comparing these choices with the supplied reference project.
- **Testing:** drafting parser/decoding tests and running lint, type checking,
  model generation, and public/private grading during development. The current
  stale-test status is disclosed above.
- **Documentation and tooling:** drafting function/class docstrings in the
  requested style, documenting changes, adjusting `make lint` to target `src`,
  and preparing this README with checked reference links.

AI proposed and applied code changes; its role was not limited to explaining
concepts or proofreading. The implementation also runs a local LLM to select
functions and argument values, which is separate from AI development assistance.
