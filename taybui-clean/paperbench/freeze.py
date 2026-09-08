"""Freeze the tested protocol before running either held-out test set."""
import argparse
import json
from pathlib import Path

from .common import atomic_json
from .prompts import METHODS
from .report import write_report
from .run import PROTOCOL_KEYS, source_hash, validate_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_runs/protocol.lock.json"))
    args = parser.parse_args()
    protocol, datasets = None, set()
    evidence = []
    for directory in args.dev_runs:
        report = write_report(directory)
        config = json.loads((directory / "manifest.json").read_text())["config"]
        if config["phase"] != "dev" or not report["completed"] or config["n"] < 96:
            raise ValueError("Freeze requires complete dev runs of at least 96 examples per dataset")
        required = set(METHODS) if config["symplan_variant"] == "full" else {"SymPlan"}
        if set(config["methods"]) != required:
            raise ValueError("Development must include all main methods, or SymPlan alone for an ablation")
        if config["source_hash"] != source_hash():
            raise ValueError("Code changed since development; rerun dev before freezing")
        current = {k: config[k] for k in PROTOCOL_KEYS}
        if protocol is not None:
            validate_resume(protocol, current)
        protocol = current
        datasets.add(config["dataset"])
        evidence.append({"directory": str(directory.resolve()), "summary": report})
    if datasets != {"math500", "gsm8k"}:
        raise ValueError("Need independent dev evidence on both MATH and GSM8K")
    lock = {"protocol": protocol, "models": [protocol["model_id"]], "development": evidence}
    if args.output.exists():
        raise ValueError("Protocol lock already exists; use a new named protocol for new experiments")
    atomic_json(args.output, lock)
    print(args.output)


if __name__ == "__main__":
    main()
