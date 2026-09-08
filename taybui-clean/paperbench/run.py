"""Run a frozen, resumable experiment on one CUDA GPU."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from .common import atomic_json, digest, load_records, question_id
from .engine import solve
from .grading import grade
from .model import MODELS, Runner
from .prompts import METHODS

PROTOCOL_KEYS = ("precision", "input_limit", "total_tokens", "extract_tokens", "plan_tokens",
                 "code_tokens", "max_retries", "exec_timeout", "executor", "executor_image",
                 "executor_image_id", "seed", "methods", "source_hash", "data_manifest_hash",
                 "prompt_profile", "symplan_variant", "model_id", "model_revision",
                 "versions", "python", "cuda", "gpu")


def source_hash():
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and
             (p.suffix in {".py", ".json"} or p.name == "LICENSE")}
    files["Dockerfile"] = (root / "Dockerfile").read_text()
    return digest(files)


def load_data(directory, dataset, phase, limit):
    manifest = json.loads((directory / "manifest.json").read_text())
    name = f"{dataset}_{'test' if phase == 'test' else 'dev'}.jsonl"
    rows = [json.loads(line) for line in (directory / name).read_text().splitlines()]
    spec = manifest["files"][name]
    if digest(rows) != spec["sha256"] or len(rows) != spec["count"]:
        raise ValueError("Data checksum/count mismatch; regenerate in a new directory")
    if len({r["id"] for r in rows}) != len(rows) or any(question_id(r["question"]) != r["id"] for r in rows):
        raise ValueError("Duplicate or inconsistent problem IDs")
    if phase == "test":
        expected = {"math500": 500, "gsm8k": 1319}[dataset]
        if limit is not None or len(rows) != expected:
            raise ValueError("Test runs must include the complete benchmark; use --phase smoke for a pilot")
    elif limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError("Sample limit outside available dev set")
        rows = rows[:limit]
    return rows, manifest


def validate_resume(previous, current):
    if previous != current:
        keys = sorted(k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k))
        raise ValueError("Refusing to mix different runs. Changed fields: " + ", ".join(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="deepseek-coder-6.7b")
    parser.add_argument("--dataset", choices=["math500", "gsm8k"], default="math500")
    parser.add_argument("--phase", choices=["smoke", "dev", "test"], default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("data/paper"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--precision", choices=["bf16", "nf4"], default="bf16")
    parser.add_argument("--input-limit", type=int, default=8192)
    parser.add_argument("--total-tokens", type=int, default=3072)
    parser.add_argument("--extract-tokens", type=int, default=256)
    parser.add_argument("--plan-tokens", type=int, default=384)
    parser.add_argument("--code-tokens", type=int, default=None)
    parser.add_argument("--prompt-profile", choices=["controlled", "source"], default="controlled")
    parser.add_argument("--symplan-variant", choices=["full", "no-extract", "no-plan", "code-only"], default="full")
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--exec-timeout", type=int, default=15)
    parser.add_argument("--executor", choices=["docker", "process"], default="docker")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, help="Frozen protocol required for test runs")
    parser.add_argument("--revision", help="Optional pinned Hugging Face model commit")
    parser.add_argument("--check-only", action="store_true", help="Validate data/config without loading a model")
    args = parser.parse_args()
    if args.code_tokens is None:
        args.code_tokens = args.total_tokens if args.prompt_profile == "source" else 1536
    if args.max_retries is None:
        args.max_retries = 0 if args.prompt_profile == "source" else 1
    if args.prompt_profile == "source" and args.max_retries != 0:
        parser.error("Source profile requires --max-retries 0")
    if any(getattr(args, k) <= 0 for k in ["input_limit", "total_tokens", "extract_tokens", "plan_tokens", "code_tokens", "exec_timeout"]):
        parser.error("Token limits and timeout must be positive")
    if args.max_retries < 0 or args.total_tokens <= args.extract_tokens + args.plan_tokens:
        parser.error("Invalid retries or insufficient budget for all three stages")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate methods")
    if args.phase == "smoke" and args.limit is None:
        args.limit = 8
    rows, data_manifest = load_data(args.data_dir, args.dataset, args.phase, args.limit)
    config = {k: getattr(args, k) for k in ["dataset", "phase", "precision", "input_limit", "total_tokens",
              "extract_tokens", "plan_tokens", "code_tokens", "max_retries", "exec_timeout", "seed", "methods",
              "prompt_profile", "symplan_variant"]}
    config.update(model_id=MODELS[args.model], n=len(rows), dataset_hash=digest(rows),
                  source_hash=source_hash(), data_manifest_hash=digest(data_manifest),
                  executor=args.executor, executor_image="symplan-executor:v1")
    if args.phase == "test" and args.lock is None:
        parser.error("Test requires --lock from completed development runs")
    if args.check_only:
        print(json.dumps(config, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU on this host. Use the RTX 3090 machine.")
    # Infrastructure errors must abort, not be counted as model failures.
    if args.executor == "docker":
        image_result = subprocess.run(["docker", "image", "inspect", config["executor_image"],
                                   "--format", "{{.Id}}"], capture_output=True, text=True, check=True)
        config["executor_image_id"] = image_result.stdout.strip()
    else:
        # The unprivileged child must not be able to read questions, golds,
        # checkpoints or credentials from this private root-owned directory.
        root = Path(__file__).resolve().parent.parent
        if os.geteuid() != 0 or root.stat().st_uid != 0 or root.stat().st_mode & 0o077:
            raise RuntimeError("For process execution, run as root and chmod 700 the project directory")
        if not args.data_dir.resolve().is_relative_to(root) or not args.output.resolve().is_relative_to(root):
            raise ValueError("Process executor requires data and output inside the private project directory")
        config["executor_image_id"] = "process-seccomp-v1"
    from .execution import execute
    health = execute('print(r"\\boxed{2}")', timeout=args.exec_timeout, backend=args.executor)
    if health["status"] != "success" or health["answer"] != "2":
        raise RuntimeError("Executor health check failed: " + str(health))
    lock = json.loads(args.lock.read_text()) if args.lock else None
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".writer.lock").open("w") as writer_lock:
        fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.output / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        from huggingface_hub import HfApi
        config["model_revision"] = args.revision or (previous["config"]["model_revision"] if previous else
                                                      (lock["protocol"]["model_revision"] if lock else
                                                       HfApi().model_info(config["model_id"]).sha))
        versions = {p: importlib.metadata.version(p) for p in
                    ["torch", "transformers", "accelerate", "math-verify", "sympy", "numpy", "scipy", "datasets",
                     "tokenizers", "huggingface-hub", "latex2sympy2-extended", "antlr4-python3-runtime"]}
        if args.precision == "nf4":
            versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        if args.executor == "process":
            versions["pyseccomp"] = importlib.metadata.version("pyseccomp")
        config["versions"] = versions
        config["python"] = platform.python_version()
        config["cuda"] = torch.version.cuda
        config["gpu"] = torch.cuda.get_device_name(0)
        if lock:
            validate_resume(lock["protocol"], {k: config[k] for k in PROTOCOL_KEYS})
        if previous:
            validate_resume(previous["config"], config)
        else:
            if (args.output / "records.jsonl").exists():
                raise ValueError("Records without a manifest; choose a clean output directory")
            atomic_json(manifest_path, {"config": config, "created_at": time.time(),
                                      "data_manifest": data_manifest})
        records = load_records(args.output / "records.jsonl")
        done = {(r["id"], r["method"]) for r in records}
        expected = {(r["id"], m) for r in rows for m in args.methods}
        if not done <= expected:
            raise ValueError("Checkpoint contains foreign problems or methods")
        if done == expected:
            from .report import write_report
            write_report(args.output)
            print("Already complete.")
            return
        runner = Runner(config["model_id"], config["model_revision"], args.precision, args.input_limit, args.seed)
        live = {method: {"correct": 0, "n": 0} for method in args.methods}
        for previous in records:
            live[previous["method"]]["correct"] += int(previous["correct"])
            live[previous["method"]]["n"] += 1

        def write_live(problem_index=None):
            snapshot = {
                "dataset": args.dataset, "phase": args.phase, "model_id": config["model_id"],
                "precision": config["precision"], "methods": list(args.methods),
                "completed_records": sum(row["n"] for row in live.values()),
                "expected_records": len(rows) * len(args.methods),
                "problem_index": problem_index,
                "accuracy": {
                    method: (100.0 * row["correct"] / row["n"] if row["n"] else None)
                    for method, row in live.items()
                },
            }
            atomic_json(args.output / "live_accuracy.json", snapshot)

        rng = random.Random(args.seed)
        ordered_rows = list(rows)
        rng.shuffle(ordered_rows)
        try:
            with (args.output / "records.jsonl").open("a") as output:
                for index, row in enumerate(ordered_rows):
                    # Counterbalance method order to reduce thermal/order timing bias.
                    offset = index % len(args.methods)
                    methods = args.methods[offset:] + args.methods[:offset]
                    for method in methods:
                        if (row["id"], method) in done:
                            continue
                        result = solve(row["question"], method, runner, config)
                        grading = grade(result["prediction"], row["answer"])
                        first = result["attempts"][0]
                        first_grading = (grading if len(result["attempts"]) == 1 else
                                         grade(first["answer"], row["answer"]))
                        if any(g["status"] in {"grader_error", "gold_parse_failure"}
                               for g in [grading, first_grading]):
                            raise RuntimeError("Grading failure; fix before continuing: " + str(grading))
                        record = {**row, "method": method, **result,
                                  "correct": grading["correct"], "grading_status": grading["status"],
                                  "first_attempt_correct": first_grading["correct"],
                                  "first_attempt_grading_status": first_grading["status"],
                                  "grading_error": grading.get("error"), "finished_at": time.time()}
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        os.fsync(output.fileno())
                        done.add((row["id"], method))
                        live[method]["correct"] += int(record["correct"])
                        live[method]["n"] += 1
                        write_live(index + 1)
                        print(f"{index+1}/{len(rows)} {method}: correct={record['correct']} "
                              f"tokens={result['output_tokens']} status={result['execution_status']} "
                              f"dataset={args.dataset} accuracy={live[method]['correct']}/{live[method]['n']}", flush=True)
                    if all((row["id"], method) in done for method in args.methods for row in [row]):
                        totals = ", ".join(
                            f"{method}={live[method]['correct']}/{live[method]['n']}"
                            for method in args.methods
                        )
                        print(f"PROBLEM_DONE dataset={args.dataset} problem={index+1}/{len(rows)} {totals}", flush=True)
        finally:
            from .report import write_report
            write_report(args.output)


if __name__ == "__main__":
    main()
