# SymCode / SymPlanner Benchmark Workspace

Main implementation lives in [`kaggle/`](./kaggle).

Read the detailed method documentation here:

- [`kaggle/README.md`](./kaggle/README.md)

Current method layout:

```text
kaggle/method/
├── direct/       # Direct final-answer baseline
├── cot/          # Chain-of-Thought baseline
├── symcode/      # Monolithic SymPy code generation
├── symplanner/   # Extract -> Plan -> SymCode generation
├── evaluator.py  # Shared benchmark loops
├── extractor.py
├── sandbox.py
├── verifier.py
└── target_contract.py
```

Quick check:

```bash
python3 kaggle/tests/test_symplanner_quality.py
python3 -m compileall -q kaggle/method kaggle/run_benchmark.py
```

Quick benchmark:

```bash
python3 kaggle/run_benchmark.py --dataset math500 --num-samples 5 --methods Direct CoT SymCode SymPlanner
```
