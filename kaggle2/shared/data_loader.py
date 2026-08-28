"""
Data loading and filtering utilities for MATH-500 and GSM8K.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional


def parse_difficulty_level(val: Any) -> Optional[int]:
    """Chuyen doi muc do kho tu so nguyen, chuoi (vi du: 'Level 2'), hoac None."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).lower().replace("level", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def load_dataset_file(
    file_path: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    filter_levels: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Nap tap du lieu JSONL voi bo loc tuy chon theo do kho Level 1-5 va so luong mau.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File du lieu khong ton tai: {file_path}")

    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("problem") or item.get("question") or ""
            a = item.get("answer") or item.get("solution") or ""
            subj = item.get("subject") or item.get("type") or "unknown"
            raw_level = item.get("level")
            lvl = parse_difficulty_level(raw_level)

            if filter_levels is not None and len(filter_levels) > 0:
                if lvl is None or lvl not in filter_levels:
                    continue

            dataset.append({
                "question": q,
                "problem": q,
                "answer": a,
                "solution": a,
                "raw": a,
                "subject": subj,
                "level": lvl,
                "level_label": f"Level {lvl}" if lvl is not None else "N/A"
            })

    if num_samples is not None and num_samples > 0:
        dataset = dataset[:num_samples]

    print(f"[INFO] Da nap {len(dataset)} mau tu '{file_path}' (Loc do kho: {filter_levels}, Gioi han mau: {num_samples})")
    return dataset
