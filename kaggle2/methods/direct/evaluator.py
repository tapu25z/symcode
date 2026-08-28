"""
Evaluator for Direct Zero-shot answering.
"""

from __future__ import annotations

import os
import gc
import json
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from shared.extractor import extract_answer_fallback, extract_ground_truth, check_exact_match
from .prompts import build_direct_messages


def _load_checkpoint(checkpoint_file: Optional[str]) -> tuple[dict, set]:
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        return {}, set()
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {}).get("Direct", [])
        return data, {r["problem"] for r in results}
    except Exception:
        return {}, set()


def _save_checkpoint(checkpoint_file: Optional[str], results: List[Dict[str, Any]]) -> None:
    if not checkpoint_file:
        return
    data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data.setdefault("results", {})["Direct"] = results
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, checkpoint_file)


def evaluate_direct(
    dataset: List[Dict[str, Any]],
    llm: Any,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5
) -> List[Dict[str, Any]]:
    """
    Danh gia phuong phap Direct Zero-shot.
    """
    existing_data, completed = _load_checkpoint(checkpoint_file)
    results = existing_data.get("results", {}).get("Direct", [])

    if completed:
        print(f"[INFO] Tiep tuc Direct: da hoan thanh {len(completed)}/{len(dataset)} mau.")

    print(f"\n==================== Bat dau danh gia Phuong phap: Direct ====================")

    new_count = 0
    for item in tqdm(dataset, desc="Danh gia Direct"):
        question = item["question"]
        if question in completed:
            continue

        gt = extract_ground_truth(item.get("raw") or item["answer"])
        messages = build_direct_messages(question)

        raw_output, token_count = llm.generate_chat(messages)
        predicted_ans = extract_answer_fallback(raw_output)
        is_correct = check_exact_match(predicted_ans, gt)

        results.append({
            "problem": question,
            "subject": item.get("subject", "unknown"),
            "level": item.get("level"),
            "level_label": item.get("level_label", "N/A"),
            "ground_truth": gt,
            "predicted": predicted_ans,
            "is_correct": is_correct,
            "generated_tokens": token_count,
            "attempts": 1,
            "execution_status": "not_applicable",
            "verification_status": "not_applicable",
            "verification_feedback": None,
            "raw_output": raw_output
        })
        completed.add(question)
        new_count += 1

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if checkpoint_file and (new_count % max(1, save_every) == 0):
            gc.collect()
            _save_checkpoint(checkpoint_file, results)

    if checkpoint_file:
        _save_checkpoint(checkpoint_file, results)

    return results
