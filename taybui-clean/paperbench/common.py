import hashlib
import json
import os
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def question_id(question):
    return hashlib.sha256(" ".join(question.split()).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def boxed(text):
    """Only the final complete boxed answer, never numbers inside a derivation."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text or ""))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                answer = text[start:i].strip()
                return answer if answer else None
    return None


def python_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    # Same extraction for all methods; no method-specific AST repairs.
    return blocks[-1].strip() if blocks else text


def load_records(path):
    if not Path(path).exists():
        return []
    # Fail loudly on partial writes instead of silently losing experiment data.
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    keys = [(r["id"], r["method"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate problem/method records in checkpoint")
    return records
