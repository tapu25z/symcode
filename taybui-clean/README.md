# Kaggle Math Reasoning Benchmark

Benchmark suite for MATH-500 and GSM8K with four methods:

1. Direct
2. Chain-of-Thought
3. SymCode
4. SymPlanner

The code is organized so each method has its own folder, while shared execution, extraction, verification, dataset loading, and metrics stay in common modules.

## Folder Structure

```text
kaggle/
├── data/
│   ├── gsm8k/test.jsonl
│   └── math500/test.jsonl
├── method/
│   ├── direct/
│   │   ├── prompt.py
│   │   └── evaluator.py
│   ├── cot/
│   │   ├── prompt.py
│   │   └── evaluator.py
│   ├── symcode/
│   │   ├── prompt.py
│   │   └── evaluator.py
│   ├── symplanner/
│   │   ├── prompt.py
│   │   └── evaluator.py
│   ├── evaluator.py
│   ├── extractor.py
│   ├── model.py
│   ├── sandbox.py
│   ├── static_lint.py
│   ├── target_contract.py
│   └── verifier.py
├── tests/test_symplanner_quality.py
├── inference.ipynb
├── requirements.txt
└── run_benchmark.py
```

The old flat imports still work:

```python
from method import evaluate_symplanner, evaluate_symcode
```

Method-folder imports also work:

```python
from method.direct import build_messages
from method.symplanner import build_extract_messages, build_planner_messages, build_codegen_messages
```

## Shared Protocol

All methods use the same dataset loader, result format, and exact-match scorer.

Shared modules:

- `method/model.py`: loads the Hugging Face model with optional 4-bit NF4 quantization.
- `method/extractor.py`: extracts boxed answers, extracts Python code, normalizes answers, and checks exact match.
- `method/sandbox.py`: executes generated Python/SymPy code with timeout.
- `method/verifier.py`: lightweight verifier for invalid outputs, target mismatch, unresolved symbols, domain issues, and known strategy mistakes.
- `method/static_lint.py`: cheap diagnostics for generated code hazards.
- `method/evaluator.py`: benchmark loops and summary metrics.

Program-based methods use the same final execution loop:

1. Extract Python code from the model response.
2. Execute it in the sandbox.
3. Extract the final `\boxed{...}` answer from stdout.
4. Run lightweight verifier when possible.
5. Retry with traceback/verifier feedback when there is a concrete failure.
6. Score with `check_exact_match(predicted, ground_truth)`.

## Method 1: Direct

Folder:

```text
method/direct/
```

Goal: direct final-answer baseline.

System prompt:

```text
You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \boxed{answer}.
```

User prompt:

```text
Problem:
{question}
```

Flow:

1. Generate one answer.
2. Extract the final boxed answer.
3. Score exact match.

No code execution and no retry loop.

## Method 2: Chain-of-Thought

Folder:

```text
method/cot/
```

Goal: natural-language reasoning baseline.

System prompt:

```text
You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \boxed{answer}.
```

User prompt:

```text
Problem:
{question}
```

Flow:

1. Generate one step-by-step solution.
2. Extract the final boxed answer.
3. Score exact match.

No code execution and no retry loop.

## Method 3: SymCode

Folder:

```text
method/symcode/
```

Goal: monolithic Python/SymPy program generation.

System prompt summary:

```text
You are an expert mathematical solver and deterministic Python/SymPy code generator.

Return only executable Python code in one ```python ... ``` block.
Import sympy as sp, formulate the problem, solve the requested target,
use exact arithmetic when possible, avoid unbounded loops, guard fragile solvers,
and print only the final LaTeX boxed answer.
```

User prompt:

```text
# PROBLEM
{question}
# END PROBLEM

Return executable Python code only enclosed in ```python ... ```.
```

Required output:

```python
print(f"\\boxed{{{final_answer}}}")
```

Flow:

1. Problem -> SymPy code.
2. Sandbox execution.
3. Boxed-answer extraction.
4. Verifier/static diagnostics.
5. Retry with failed code + traceback/diagnosis.
6. Exact-match scoring.

## Method 4: SymPlanner

Folder:

```text
method/symplanner/
```

Goal: keep SymCode simple, but reduce formulation errors by adding two short model turns before code generation.

SymPlanner is intentionally simple:

```text
Problem
  -> Extract mathematical state
  -> Generate solution plan without final answer
  -> Generate SymCode from problem + extraction + plan
  -> Execute/retry exactly like SymCode
```

### Turn 1: Extract

System prompt:

```text
You extract the mathematical state of a problem for a later solver.

Return ONLY these labeled lines:
# Target: quantity/expression/object the problem asks for
# Given: facts, numbers, definitions, equations, and relations
# Constraints: domains, integer/positive/nonzero/range/order conditions
# Output: number|symbolic|tuple|set|matrix|text|base_notation

Rules:
- Do not solve the problem.
- Do not write code.
- Keep each line short and factual.
```

User prompt:

```text
# PROBLEM
{question}

Extract the mathematical state only.
```

Example extract:

```text
# Target: number of distinct expression values
# Given: expression 2*3*4*5+1; terms cannot be rearranged
# Constraints: only parentheses may be inserted
# Output: number
```

### Turn 2: Plan

System prompt:

```text
You write a short solution plan for a Python/SymPy solver.

You will receive the original problem and an extracted mathematical state.
Return ONLY numbered plan steps.

Rules:
- Do not calculate or reveal the final numeric answer.
- Do not write Python code.
- Include candidate filtering or constraint checks when needed.
- Keep the plan short.
```

User prompt:

```text
# PROBLEM
{question}

# EXTRACTED STATE
{extract}

Write the plan only.
```

Example plan:

```text
1. Represent the ordered numbers and operators.
2. Enumerate all valid parenthesizations recursively.
3. Collect unique evaluated values.
4. Return the number of unique values.
```

### Turn 3: SymCode Generation

System prompt summary:

```text
You are an expert mathematical solver and deterministic Python/SymPy code generator.

Return only executable Python code in one ```python ... ``` block.
Use the problem, extraction, and plan. Solve the requested target.
Print only the final answer in LaTeX boxed format.
```

User prompt:

```text
# PROBLEM
{question}

# EXTRACTED STATE AND PLAN
# EXTRACTED STATE
{extract}

# PLAN
{plan}

# OUTPUT REQUIREMENT
- Answer type: ...
- Unit: ...
- Diagram relations required: ...

Return executable Python code only enclosed in ```python ... ```. Do not write explanations.
```

Required output is the same as SymCode:

```python
print(f"\\boxed{{{final_answer}}}")
```

After code generation, SymPlanner uses the same sandbox, boxed-answer extractor, verifier, static lint, retry, and exact-match scorer as SymCode. The only experimental difference is the added Extract + Plan context before codegen.

## Output JSON

Each benchmark output file contains:

```json
{
  "config": {
    "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset_name": "math500",
    "methods_to_run": ["Direct", "CoT", "SymCode", "SymPlanner"]
  },
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "results": {
    "Direct": [],
    "CoT": [],
    "SymCode": [],
    "SymPlanner": []
  },
  "live_accuracy": {
    "Direct": {
      "accuracy_percent": 80.0,
      "correct": 4,
      "total": 5
    }
  },
  "live_accuracy_by_problem": [],
  "summary": {}
}
```

SymPlanner rows additionally include:

- `extraction_note`
- `planner_note`
- `symplanner_context`
- `attempt_history`
- generated code and execution diagnostics

## Running

Install:

```bash
pip install -r requirements.txt
```

Quick run. By default, the benchmark now runs one problem at a time and evaluates each selected method before moving to the next problem:

```bash
python3 run_benchmark.py --dataset math500 --num-samples 5 --methods Direct CoT SymCode SymPlanner
```

Default run order:

```text
Problem 1 -> Direct -> CoT -> SymCode -> SymPlanner
Problem 2 -> Direct -> CoT -> SymCode -> SymPlanner
...
```

During the run, each method reports whether it got the current problem right. After all selected methods finish that problem, the script prints one live accuracy snapshot:

```text
[PROBLEM 5/50] ...
[Direct] cau nay: DUNG | pred=42 | gt=42
[CoT] cau nay: SAI | pred=40 | gt=42
[LIVE ACC PROBLEM 5/50] Direct: 60.00% (3/5) | CoT: 80.00% (4/5)
```

The latest totals are saved into `live_accuracy`, and the per-problem snapshots are saved into `live_accuracy_by_problem`, so you can inspect the output JSON while the benchmark is still running.

Run in the old order, one full method at a time:

```bash
python3 run_benchmark.py --dataset math500 --num-samples 5 --run-order by-method
```

Run only SymPlanner:

```bash
python3 run_benchmark.py --dataset math500 --methods SymPlanner --output-file symplanner_math500_full.json
```

Run selected levels:

```bash
python3 run_benchmark.py --dataset math500 --filter-levels 4 5 --methods SymCode SymPlanner
```

Run a fixed number of samples per level:

```bash
python3 run_benchmark.py --dataset math500 --filter-levels 1 2 3 4 5 --per-level-samples 20 --methods Direct CoT SymCode SymPlanner
```

Resume from an existing output file:

```bash
python3 run_benchmark.py --dataset math500 --num-samples 50 --output-file math500_50_results.json
```

If the output file already contains results for a problem/method pair, the runner skips that pair and continues from the checkpoint.

Use a local JSONL dataset:

```bash
python3 run_benchmark.py --dataset-path data/math500/test.jsonl --methods Direct SymPlanner
```

Important options:

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--dataset` | `math500` | `math500` or `gsm8k` |
| `--methods` | all four | Methods to evaluate |
| `--run-order` | `by-problem` | `by-problem` runs each selected method per problem; `by-method` runs the old method-by-method flow |
| `--num-samples` | all | Limit number of samples |
| `--filter-levels` | none | Evaluate selected levels |
| `--per-level-samples` | none | Select N samples per level |
| `--tail` | false | Select from the end of each list |
| `--model-id` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Hugging Face model id |
| `--load-in-4bit` | true | Use 4-bit NF4 quantization |
| `--no-4bit` | false | Disable 4-bit loading |
| `--max-new-tokens` | `1024` | Max generation tokens per model call |
| `--temperature` | `0.0` | Greedy decoding by default |
| `--max-retries` | `2` | Retry count for program methods |
| `--timeout` | `15` | Sandbox timeout in seconds |
| `--save-every` | `5` | Checkpoint interval |

## Customizing Methods

You can choose which built-in methods to run with `--methods`:

```bash
python3 run_benchmark.py --dataset math500 --num-samples 10 --methods Direct SymPlanner
```

The valid built-in names are:

```text
Direct CoT SymCode SymPlanner
```

To change the behavior of an existing method, edit its prompt file:

```text
method/direct/prompt.py
method/cot/prompt.py
method/symcode/prompt.py
method/symplanner/prompt.py
```

To add a brand-new method name, add a new folder under `method/`, implement its `prompt.py` and `evaluator.py`, then register it in:

```text
run_benchmark.py
method/__init__.py
```

The CLI currently validates method names with fixed `choices`, so a new custom method will not run from `--methods` until it is added to those choices and to the dispatch logic.

## Tests

```bash
pytest -q
python3 -m compileall -q method run_benchmark.py
```

Avoid `python -m unittest kaggle.tests...` from a parent repository root on machines with the Kaggle API installed, because Python may resolve `kaggle` to the external package.

## Paper Note

The paper should describe the current method as:

```text
SymPlanner: Extract-then-Plan guided SymCode generation.
```

The clean method equation is:

```text
E = Extract_M(q)
P = Plan_M(q, E)
C = Codegen_M(q, E, P)
y = Exec(C)
```

Everything after `C` is the same execution and repair path as SymCode.
