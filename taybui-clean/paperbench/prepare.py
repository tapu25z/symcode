"""Download complete official test sets and independent training-derived dev sets."""
import argparse
import collections
import json
import random
from pathlib import Path

from .common import atomic_json, boxed, digest, question_id

SOURCES = {
    "math500_test": ("HuggingFaceH4/MATH-500", "default", "test", 500),
    "math500_dev": ("DigitalLearningGmbH/MATH-lighteval", "default", "train", 7500),
    "gsm8k_test": ("openai/gsm8k", "main", "test", 1319),
    "gsm8k_dev": ("openai/gsm8k", "main", "train", 7473),
}


def normalize(row, math_dataset):
    question = row.get("problem", row.get("question"))
    if math_dataset:
        answer = row.get("answer") or boxed(row["solution"])
    else:
        answer = row["answer"].split("####")[-1].strip()
    if not question or not answer:
        raise ValueError("Missing question or reference answer")
    return {"id": question_id(question), "question": question, "answer": answer,
            "subject": row.get("subject", row.get("type", "Arithmetic")),
            "level": str(row.get("level", "NA")),
            "source_id": row.get("unique_id")}


def choose_dev(rows, excluded, count, seed):
    # Round-robin over shuffled subject/level groups, independently of accuracy.
    groups = collections.defaultdict(list)
    seen = set(excluded)
    for row in rows:
        if row["id"] not in seen:
            groups[row["subject"], row["level"]].append(row)
            seen.add(row["id"])
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    if len(selected) != count:
        raise ValueError("Insufficient disjoint development examples")
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/paper"))
    parser.add_argument("--dev-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dev_size < 1:
        parser.error("--dev-size must be positive")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["dev_size"] != args.dev_size or manifest["seed"] != args.seed:
            raise ValueError("Existing data uses a different split; choose a new output directory")
        for name, spec in manifest["files"].items():
            rows = [json.loads(x) for x in (args.output / name).read_text().splitlines()]
            if digest(rows) != spec["sha256"]:
                raise ValueError(f"Dataset modified: {name}")
        print("Dataset manifest and all checksums verified.")
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "dev_size": args.dev_size, "files": {}}
    revisions = {}
    datasets = {}
    for name, (repo, config, split, expected) in SOURCES.items():
        revision = revisions.setdefault(repo, HfApi().dataset_info(repo).sha)
        raw = load_dataset(repo, config, split=split, revision=revision)
        if len(raw) != expected:
            raise ValueError(f"{name}: expected {expected}, found {len(raw)}")
        rows, invalid_references = [], 0
        for r in raw:
            try:
                rows.append(normalize(r, name.startswith("math")))
            except ValueError:
                if name.endswith("test"):
                    raise
                invalid_references += 1
        if len({r["id"] for r in rows}) != len(rows) and name.endswith("test"):
            raise ValueError(f"Duplicate test problems: {name}")
        if name.endswith("dev"):
            test = datasets[name.replace("dev", "test")]
            rows = choose_dev(rows, {r["id"] for r in test}, args.dev_size, args.seed)
        datasets[name] = rows
        filename = name + ".jsonl"
        (args.output / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["files"][filename] = {"repo": repo, "revision": revision,
            "config": config, "source_split": split, "source_count": expected,
            "count": len(rows), "invalid_train_references_excluded": invalid_references,
            "sha256": digest(rows)}
        print(f"{filename}: {len(rows)} questions, revision {revision}", flush=True)
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
