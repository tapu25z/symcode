"""Recompute all metrics on the same IDs, including unfinished checkpoints."""
import argparse
import collections
import json
import math
from pathlib import Path

from .common import atomic_json, load_records
from .prompts import BASELINES


def paired(a, b):
    import numpy as np
    ids = sorted(a.keys() & b.keys())
    wins = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
    losses = sum(b[i]["correct"] and not a[i]["correct"] for i in ids)
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) /
            2**discordant) if discordant else 1.
    n = len(ids)
    rng = np.random.default_rng(42)
    draws = rng.multinomial(n, [wins/n, losses/n, 1-(wins+losses)/n], size=10000)
    low, high = np.percentile(100*(draws[:, 0]-draws[:, 1])/n, [2.5, 97.5])
    return {"n": n, "wins": wins, "losses": losses, "delta_pp": 100*(wins-losses)/n,
            "mcnemar_exact_p": p, "paired_bootstrap_95ci_pp": [float(low), float(high)]}


def summarize(records, methods, expected):
    grouped = {m: {r["id"]: r for r in records if r["method"] == m} for m in methods}
    common = set.intersection(*(set(g) for g in grouped.values()))
    n = len(common)
    result = {"expected": expected, "common_n": n,
              "completed": all(len(g) == expected for g in grouped.values()),
              "available": {m: len(g) for m, g in grouped.items()}, "methods": {}, "paired": {}}
    if not n:
        return result
    for method, group in grouped.items():
        rows = [group[i] for i in sorted(common)]
        correct = sum(r["correct"] for r in rows)
        result["methods"][method] = {"correct": correct, "n": n,
            "accuracy": 100*correct/n,
            "first_attempt_accuracy": (100*sum(r["first_attempt_correct"] for r in rows)/n
                                       if all("first_attempt_correct" in r for r in rows) else None),
            "esr": 100*sum(r["execution_status"] == "success" for r in rows)/n,
            "first_attempt_esr": 100*sum(bool(r["attempts"]) and r["attempts"][0]["status"] == "success" for r in rows)/n,
            "avg_output_tokens": sum(r["output_tokens"] for r in rows)/n,
            "avg_input_tokens": sum(r["input_tokens"] for r in rows)/n,
            "avg_calls": sum(len(r["calls"]) for r in rows)/n,
            "avg_solve_seconds": sum(r["solve_seconds"] for r in rows)/n,
            "truncated_call_count": sum(r["limit_hits"] for r in rows),
            "grading_statuses": dict(collections.Counter(r["grading_status"] for r in rows)),
            "by_level": {}, "by_subject": {}}
        for field in ["level", "subject"]:
            for value in sorted({r[field] for r in rows}):
                subset = [r for r in rows if r[field] == value]
                result["methods"][method]["by_" + field][value] = {
                    "n": len(subset), "correct": sum(r["correct"] for r in subset),
                    "accuracy": 100*sum(r["correct"] for r in subset)/len(subset)}
    if "SymPlan" in grouped:
        for baseline in methods:
            if baseline != "SymPlan":
                result["paired"][baseline] = paired(
                    {i: grouped["SymPlan"][i] for i in common},
                    {i: grouped[baseline][i] for i in common})
        # Correct the three predeclared main baseline comparisons, under the controlled protocol.
        comparisons = [(name, r) for name, r in result["paired"].items()
                       if name in BASELINES]
        comparisons.sort(key=lambda item: item[1]["mcnemar_exact_p"])
        previous = 0.
        for rank, (_, row) in enumerate(comparisons):
            previous = max(previous, min(1., (len(comparisons)-rank)*row["mcnemar_exact_p"]))
            row["holm_adjusted_p"] = previous
    return result


def write_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    report = summarize(load_records(directory / "records.jsonl"), config["methods"], config["n"])
    report["phase"] = config["phase"]
    report["model_id"] = config["model_id"]
    report["dataset"] = config["dataset"]
    atomic_json(directory / "summary.json", report)
    lines = [f"# {config['model_id']} / {config['dataset']} / {config['phase']}", "",
             f"Common completed problems: {report['common_n']}/{report['expected']}. " +
             ("Complete." if report["completed"] else "PARTIAL CHECKPOINT; not a final paper result."), "",
             f"Profile: {config.get('prompt_profile', 'controlled')}; SymPlan variant: {config.get('symplan_variant', 'full')}. "
             "Controlled = matched zero-shot + repair; source = source-aligned prompts, PaL 8-shot, no repair. "
             "These are adaptations to the evaluated model, not original-paper score reproductions; see METHOD_RESEARCH.md.", "",
             "| Method | Correct/N | Accuracy % | First attempt accuracy % | ESR % | Output tokens | Input tokens | Calls | Seconds/problem |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["methods"].items():
        esr = "—" if row["esr"] is None else f"{row['esr']:.2f}"
        first = row["first_attempt_accuracy"]
        first_text = "—" if first is None else f"{first:.2f}"
        lines.append(f"| {name} | {row['correct']}/{row['n']} | {row['accuracy']:.2f} | {first_text} | {esr} | "
                     f"{row['avg_output_tokens']:.1f} | {row['avg_input_tokens']:.1f} | "
                     f"{row['avg_calls']:.2f} | {row['avg_solve_seconds']:.2f} |")
    lines += ["", "Paired comparisons: SymPlan minus each baseline; bootstrap intervals are unadjusted.", ""]
    for name, row in report["paired"].items():
        low, high = row["paired_bootstrap_95ci_pp"]
        lines.append(f"- {name}: {row['delta_pp']:+.2f} pp, 95% CI [{low:.2f}, {high:.2f}], "
                     f"wins/losses {row['wins']}/{row['losses']}, p={row['mcnemar_exact_p']:.5f}" +
                     (f", Holm p={row['holm_adjusted_p']:.5f}" if "holm_adjusted_p" in row else ""))
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = write_report(args.run)
    print(f"{args.run / 'report.md'}: {report['common_n']}/{report['expected']} common problems")


if __name__ == "__main__":
    main()
