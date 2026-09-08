"""Fixed, resumable paper suite: smoke, dev, freeze, full test, table export."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .common import atomic_json
from .model import MODELS

# Predeclared; no winner gate, test-based tuning, or selective reporting.
SUITES = {
    "controlled": ["--prompt-profile", "controlled", "--max-retries", "1"],
    "source": ["--prompt-profile", "source", "--max-retries", "0", "--code-tokens", "3072"],
    **{name: ["--prompt-profile", "controlled", "--symplan-variant", name,
              "--methods", "SymPlan", "--max-retries", "1"]
       for name in ["no-extract", "no-plan", "code-only"]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen3-4b")
    parser.add_argument("--output", type=Path, default=Path("paper_runs/paper_v3"))
    parser.add_argument("--executor", choices=["process", "docker"], default="process")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Lock the entire supervisor, including between dataset subprocesses.
    import fcntl
    with (args.output / ".pipeline.lock").open("w") as pipeline_lock:
        fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_suite(args)


def run_suite(args):
    def status(state, **fields):
        atomic_json(args.output / "status.json", {"state": state, "model": args.model,
                    "updated_at": time.time(), **fields})
        print(state, fields, flush=True)

    settings_file = args.output / "suite.json"
    from .run import source_hash, validate_resume
    settings = {"model": args.model, "executor": args.executor,
                "suites": args.suites, "source_hash": source_hash()}
    if settings_file.exists():
        saved = json.loads(settings_file.read_text())
        validate_resume(saved["settings"], settings)
        revision = saved["revision"]
    else:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(MODELS[args.model]).sha
        atomic_json(settings_file, {"settings": settings, "revision": revision})

    def run(phase, dataset, directory, flags, lock=None):
        command = [sys.executable, "-m", "paperbench.run", "--model", args.model,
                   "--revision", revision, "--dataset", dataset, "--phase", phase,
                   "--executor", args.executor, "--output", str(directory), *flags]
        if lock:
            command += ["--lock", str(lock)]
        subprocess.run(command, check=True)

    try:
        for name in args.suites:
            flags = SUITES[name]
            directory = args.output / name
            status("smoke", suite=name)
            run("smoke", "math500", directory / "smoke", flags)
            dev_runs = [directory / f"{d}_dev" for d in ["math500", "gsm8k"]]
            for dataset, output in zip(["math500", "gsm8k"], dev_runs):
                status("development", suite=name, dataset=dataset)
                run("dev", dataset, output, flags)
            lock = directory / "protocol.lock.json"
            if not lock.exists():
                subprocess.run([sys.executable, "-m", "paperbench.freeze", "--dev-runs",
                                *(str(p) for p in dev_runs), "--output", str(lock)], check=True)
            for dataset in ["math500", "gsm8k"]:
                status("test", suite=name, dataset=dataset,
                       note="Fixed protocol: results do not change subsequent runs.")
                run("test", dataset, directory / f"{dataset}_test", flags, lock)
        status("exporting")
        subprocess.run([sys.executable, "-m", "paperbench.export", str(args.output)], check=True)
        status("complete", tables=str(args.output / "tables"))
    except Exception as exc:
        status("error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
