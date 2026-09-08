"""Export complete held-out runs to CSV/LaTeX without inventing or selecting scores."""
import argparse
import csv
import json
from pathlib import Path

from .common import atomic_json, load_records
from .report import paired, write_report
from .run import PROTOCOL_KEYS, source_hash, validate_resume


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(root):
    root = Path(root)
    suite = json.loads((root / "suite.json").read_text())
    if suite["settings"]["source_hash"] != source_hash():
        raise ValueError("Code changed since the suite; export using the frozen code")
    output = root / "tables"
    metrics, comparisons, manifests, records = [], [], {}, {}
    for name in suite["settings"]["suites"]:
        for dataset in ["math500", "gsm8k"]:
            directory = root / name / f"{dataset}_test"
            config = json.loads((directory / "manifest.json").read_text())["config"]
            if config["source_hash"] != suite["settings"]["source_hash"] or config["model_revision"] != suite["revision"]:
                raise ValueError("Test run does not belong to the frozen suite")
            lock = json.loads((root / name / "protocol.lock.json").read_text())
            validate_resume(lock["protocol"], {k: config[k] for k in PROTOCOL_KEYS})
            if config["phase"] != "test" or config["n"] != {"math500": 500, "gsm8k": 1319}[dataset]:
                raise ValueError("Only full test runs can be exported as paper tables")
            rows = load_records(directory / "records.jsonl")
            report = write_report(directory)
            if not report["completed"] or report["common_n"] != config["n"]:
                raise ValueError(f"Incomplete test run: {directory}")
            if any(r["grading_status"] in {"grader_error", "gold_parse_failure"} or
                   r.get("first_attempt_grading_status") in {"grader_error", "gold_parse_failure"}
                   for r in rows):
                raise ValueError(f"Resolve grading errors before paper export: {directory}")
            manifests[name, dataset], records[name, dataset] = config, rows
            for method, row in report["methods"].items():
                metrics.append({"suite": name, "dataset": dataset, "method": method,
                    "model": config["model_id"], "correct": row["correct"], "n": row["n"],
                    **{k: row[k] for k in ["accuracy", "first_attempt_accuracy", "esr",
                       "first_attempt_esr", "avg_output_tokens", "avg_input_tokens",
                       "avg_calls", "avg_solve_seconds"]}})
            for baseline, row in report["paired"].items():
                comparisons.append({"suite": name, "dataset": dataset, "comparison": "SymPlan - " + baseline,
                    "n": row["n"], "delta_pp": row["delta_pp"], "wins": row["wins"], "losses": row["losses"],
                    "ci_low": row["paired_bootstrap_95ci_pp"][0], "ci_high": row["paired_bootstrap_95ci_pp"][1],
                    "p": row["mcnemar_exact_p"], "holm_p": row.get("holm_adjusted_p")})
    ablations = []
    for variant in ["no-extract", "no-plan", "code-only"]:
        for dataset in ["math500", "gsm8k"]:
            if (variant, dataset) not in records or ("controlled", dataset) not in records:
                continue
            keys = [k for k in PROTOCOL_KEYS if k not in {"methods", "symplan_variant"}]
            validate_resume({k: manifests["controlled", dataset][k] for k in keys},
                            {k: manifests[variant, dataset][k] for k in keys})
            full = {r["id"]: r for r in records["controlled", dataset] if r["method"] == "SymPlan"}
            other = {r["id"]: r for r in records[variant, dataset]}
            if full.keys() != other.keys():
                raise ValueError("Ablation problem IDs differ")
            row = paired(full, other)
            ablations.append({"dataset": dataset, "comparison": "full - " + variant,
                              "delta_pp": row["delta_pp"], "n": row["n"],
                              "ci_low": row["paired_bootstrap_95ci_pp"][0],
                              "ci_high": row["paired_bootstrap_95ci_pp"][1],
                              "p_unadjusted": row["mcnemar_exact_p"]})
    # Publish tables only after every selected test run passes validation.
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "paired.csv", comparisons)
    write_csv(output / "ablations.csv", ablations)
    lines = [r"\begin{tabular}{lllrrrr}", r"\hline",
             r"Protocol & Dataset & Method & Correct/N & Acc. (\%) & ESR (\%) & Tokens \\", r"\hline"]
    for r in metrics:
        lines.append(f"{r['suite']} & {r['dataset']} & {r['method']} & {r['correct']}/{r['n']} & "
                     f"{r['accuracy']:.2f} & {r['esr']:.2f} & {r['avg_output_tokens']:.1f}" + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (output / "results.tex").write_text("\n".join(lines) + "\n")
    atomic_json(output / "provenance.json", {"suite": suite, "runs": {
        f"{name}/{dataset}": cfg for (name, dataset), cfg in manifests.items()}})
    (output / "README.md").write_text(
        "# Paper tables\n\nFull test results only. Report all selected protocols, including losses.\n"
        "controlled: zero-shot adaptations + shared repair. source: source-aligned prompts, no repair; "
        "PaL has 8 original demonstrations and is not shot-matched. Both use the same Qwen model.\n"
        "Neither table reproduces original-paper scores. Ablation p-values and all confidence intervals "
        "are unadjusted; main baseline comparisons include Holm-adjusted p-values in paired.csv.\n"
        "Results support claims only for the evaluated model and datasets. See METHOD_RESEARCH.md.\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(export(args.root))


if __name__ == "__main__":
    main()
