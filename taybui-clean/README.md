# Math Reasoning Benchmark

Repo benchmark cac phuong phap giai toan tren MATH-500 va GSM8K:

- `Direct`: tra loi truc tiep.
- `CoT`: giai thich tung buoc bang ngon ngu tu nhien.
- `SymCode`: sinh mot chuong trinh Python/SymPy de giai bai.
- `SymPlanner`: tach bai toan thanh extract + plan + codegen, roi chay code va retry khi loi.

## Cau Truc Thu Muc

```text
.
|-- data/
|   |-- gsm8k/test.jsonl
|   `-- math500/test.jsonl
|-- method/
|   |-- direct/          # baseline Direct
|   |-- cot/             # baseline Chain-of-Thought
|   |-- symcode/         # sinh code SymPy mot luot
|   |-- symplanner/      # wrapper prompt cho SymPlanner
|   |-- evaluator.py     # vong lap benchmark, checkpoint, metrics
|   |-- extractor.py     # tach code, tach boxed answer, normalize answer
|   |-- model.py         # load model Hugging Face
|   |-- sandbox.py       # chay code sinh ra voi timeout
|   |-- static_lint.py   # chan mot so loi code re tien
|   |-- target_contract.py
|   `-- verifier.py      # verifier nhe cho output/code
|-- tests/
|   `-- test_symplanner_quality.py
|-- results/            # noi luu file ket qua benchmark
|-- requirements.txt
|-- run_benchmark.py
`-- audit_symplanner.py
```

## Cai Dat

Repo nay duoc toi gian cho workflow chay truc tiep tren may GPU thue qua CLI. Nen dung Python 3.10+ va GPU CUDA neu chay model 7B/8B cuc bo.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tat ca file ket qua benchmark nen luu trong thu muc `results/`. Thu muc nay duoc tao san va cac file JSON ben trong se khong commit len git.

Neu can tai model gated tren Hugging Face, dang nhap truoc:

```bash
huggingface-cli login
```

Mac dinh script load model o che do 4-bit NF4. Neu muon chay full precision:

```bash
python3 run_benchmark.py --no-4bit ...
```

## Cach Chay Nhanh

Smoke test 5 cau MATH-500 voi 4 method chinh:

```bash
python3 run_benchmark.py \
  --dataset math500 \
  --num-samples 5 \
  --model-preset qwen2.5-coder-7b \
  --methods Direct CoT SymCode SymPlanner \
  --output-file results/results_smoke_math500.json
```

Smoke test GSM8K:

```bash
python3 run_benchmark.py \
  --dataset gsm8k \
  --num-samples 5 \
  --model-preset qwen2.5-coder-7b \
  --methods Direct CoT SymCode SymPlanner \
  --output-file results/results_smoke_gsm8k.json
```

`run_benchmark.py` tu dong checkpoint vao `--output-file`; neu chay lai cung file, cac cau da co ket qua se duoc skip. Neu khong truyen `--output-file`, script se tu luu vao `results/`.

## Chay Full MATH-500/GSM8K Tren 3 Model

Ba preset dang co trong code:

- `qwen2.5-coder-7b`: `Qwen/Qwen2.5-Coder-7B-Instruct`
- `qwen3-8b`: `Qwen/Qwen3-8B`
- `llama3-8b`: `meta-llama/Meta-Llama-3-8B-Instruct`

Chay full MATH-500:

```bash
for model in qwen2.5-coder-7b qwen3-8b llama3-8b; do
  python3 run_benchmark.py \
    --dataset math500 \
    --model-preset "$model" \
    --methods Direct CoT SymCode SymPlanner \
    --output-file "results/results_math500_${model}.json"
done
```

Chay full GSM8K:

```bash
for model in qwen2.5-coder-7b qwen3-8b llama3-8b; do
  python3 run_benchmark.py \
    --dataset gsm8k \
    --model-preset "$model" \
    --methods Direct CoT SymCode SymPlanner \
    --output-file "results/results_gsm8k_${model}.json"
done
```

Neu muon tang so token sinh:

```bash
python3 run_benchmark.py \
  --dataset math500 \
  --model-preset qwen3-8b \
  --max-new-tokens 2048 \
  --methods Direct CoT SymCode SymPlanner \
  --output-file results/results_math500_qwen3_8b_2048.json
```

## Chay Ablation Cho SymPlanner

Cac bien the ablation:

- `SymPlanner`: full pipeline, dung ca extract va plan.
- `SymPlannerExtractOnly`: chi dung extract, bo plan.
- `SymPlannerPlanOnly`: chi dung plan, bo extract.
- `SymPlannerNoModules`: bo ca extract va plan, gan nhu pure codegen theo prompt SymPlanner.

Chay ablation MATH-500:

```bash
python3 run_benchmark.py \
  --dataset math500 \
  --model-preset qwen2.5-coder-7b \
  --methods SymPlanner SymPlannerExtractOnly SymPlannerPlanOnly SymPlannerNoModules \
  --output-file results/results_ablation_math500_qwen25.json
```

Chay ablation GSM8K:

```bash
python3 run_benchmark.py \
  --dataset gsm8k \
  --model-preset qwen2.5-coder-7b \
  --methods SymPlanner SymPlannerExtractOnly SymPlannerPlanOnly SymPlannerNoModules \
  --output-file results/results_ablation_gsm8k_qwen25.json
```

## Y Tuong Method

### SymPlanner

SymPlanner tach viec sinh code thanh cac pha nho de giam loi formulate bai toan:

```text
Problem
  -> Extract mathematical state
  -> Write short solution plan
  -> Generate Python/SymPy code from problem + extract + plan
  -> Sandbox execution
  -> Verify output
  -> Retry/debug with traceback or verifier feedback
```

Pha extract bat model noi ro target, du kien, rang buoc va kieu output. Pha plan chi viet cac buoc giai, khong tinh dap an cuoi va khong viet code. Pha codegen moi sinh Python/SymPy, in ket qua duy nhat dang `\boxed{...}`. Neu code crash, timeout, in output khong hop le, hoac verifier thay dau hieu sai target, evaluator se goi prompt debug de sua code.

### Direct

Baseline don gian nhat: model doc bai va tra ve dap an cuoi trong `\boxed{answer}`. Khong chay code, khong retry.

### CoT

Baseline reasoning bang ngon ngu tu nhien: model viet loi giai tung buoc, cuoi cung in `\boxed{answer}`. Khong chay code, khong retry.

### SymCode

Baseline program-aided: model sinh mot block Python/SymPy truc tiep tu problem. Code duoc chay trong sandbox, tach output boxed, verifier/lint kiem tra loi, va retry toi da `--max-retries` lan neu co loi cu the.

## Prompt Chi Tiet

### Direct

System:

```text
You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \boxed{answer}.
```

User:

```text
Problem:
{question}
```

### CoT

System:

```text
You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \boxed{answer}.
```

User:

```text
Problem:
{question}
```

### SymCode

System:

````text
You are an expert mathematical solver and deterministic Python/SymPy code generator.

Solve the problem by returning ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp (and math, fractions if helpful).
2. Write the solver with two independent paths (Path A: Symbolic/Analytical, Path B: Empirical/Simulation/Search loop) to cross-verify the answer whenever possible.
3. Guard symbolic solving calls (e.g., sp.solve) with try-except blocks. If SymPy fails, automatically fallback to a bounded search loop or numerical optimization.
4. Define all given quantities and formulate equations accurately.
5. Solve for the target quantity symbolically or numerically.
6. Never call `.evalf()` on standard Python int/float.
7. Avoid using sp.solve() or sp.nonlinsolve() on complex nonlinear or multivariate systems of high degree (e.g. degree >= 3 with multiple variables, or equations containing non-rational exponent powers like **(1/3)), as it causes SymPy to hang indefinitely. Use numerical optimization (e.g., scipy.optimize.minimize or fsolve) instead.
8. Never write infinite loops or unbounded while loops (e.g., custom prime generators). Always use finite for loops (e.g., for i in range(10000)) or specify a maximum iteration count to guarantee termination.
9. Print ONLY the final answer in LaTeX boxed format at the end:
   print(f"\\boxed{{{final_answer}}}")
````

User:

````text
# PROBLEM
{question}
# END PROBLEM

Return executable Python code only enclosed in ```python ... ```.
````

Debug system:

````text
You are repairing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code in one ```python ... ``` block.
Fix the reported issue and keep correct code. Do not explain or output <think> tags.

Rules:
- Recompute the target; do not hard-code an answer.
- Use exact arithmetic where possible and handle fragile solver failures.
- Use finite loops only; never use an unbounded while loop.
- Any reasoning comment must start with "# Step <number>:".
- Print only the required final result.
````

### SymPlanner

Extract system:

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

Extract user:

```text
# PROBLEM
{question}

Extract the mathematical state only.
```

Planner system:

```text
You write a short solution plan for a Python/SymPy solver.

You will receive the original problem and an extracted mathematical state.
Return ONLY numbered plan steps.

Rules:
- Do not calculate or reveal the final numeric answer.
- Do not write Python code.
- Restate the requested target when the problem asks for an input, multiplier, quotient, coefficient, vector, or tuple; do not plan to print a downstream computed value instead.
- Include candidate filtering or constraint checks when needed.
- Keep the plan short.
```

Planner user:

```text
# PROBLEM
{question}

# EXTRACTED STATE
{extraction}

Write the plan only.
```

Codegen system:

````text
You are an expert mathematical solver and deterministic Python/SymPy code generator.

Return ONLY executable Python code in one ```python ... ``` block. Do not explain.

Rules:
1. Import sympy as sp. Use exact arithmetic, especially sp.Rational; use floats only when requested.
2. Solve the requested target, not an intermediate value. Use the extraction and plan.
   Examples: if asked "by what number should A be multiplied", print the multiplier, not A times that multiplier; if asked for a quotient, print the quotient polynomial, not the remainder or value at a point.
3. If using sp.solve or another fragile solver, handle failure or an empty result. Use a simple bounded fallback only when practical.
4. Use finite loops only. Never use an unbounded while loop.
5. Add a cheap substitution or direct check when it is natural. Do not add a second algorithm just for show.
6. Never print None, Invalid, NaN, undefined variables, debug text, or intermediate values.
7. Any reasoning comment must start with "# Step <number>:".
8. At the end, print ONLY the final answer in LaTeX boxed format:
   print(f"\\boxed{{{final_answer}}}")
````

Codegen user:

````text
# PROBLEM
{question}

# EXTRACTED STATE AND PLAN
{plan_block}

# OUTPUT REQUIREMENT
- Answer type: {answer_type}
- Unit: {unit}
- Diagram relations required: {yes_or_no}

Return executable Python code only enclosed in ```python ... ```. Do not write explanations.
````

Debug user:

````text
# PROBLEM
{question}

# EXTRACTED STATE AND PLAN
{plan_block}

# OUTPUT REQUIREMENT
- Answer type: {answer_type}
- Unit: {unit}
- Diagram relations required: {yes_or_no}

# PREVIOUS CODE
```python
{bad_code}
```

# DIAGNOSIS
Execution status: {status}
Traceback:
{traceback_if_any}
Candidate answer printed: {candidate_answer_if_any}
Verifier diagnosis: {verification_feedback_if_any}

Fix the issue and return corrected executable Python code only enclosed in ```python ... ```.
````

## Ket Qua

File output JSON gom:

- `config`: model, dataset, methods, token limit, timeout.
- `results`: ket qua tung method/tung cau.
- `summary`: accuracy, token trung binh, retry/exec/verifier stats.
- `live_accuracy`: accuracy cap nhat trong luc chay.
- `live_accuracy_by_problem`: snapshot sau moi problem khi dung `--run-order by-problem`.

Chay test nhanh cho logic SymPlanner:

```bash
pytest tests/test_symplanner_quality.py
```
