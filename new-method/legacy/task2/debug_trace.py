#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verbose trace logging for Task 2 Kaggle inference debugging."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_EVENTS: list[dict[str, Any]] = []


def enabled() -> bool:
    return os.getenv("TASK2_TRACE_PROMPTS", "0").lower() in {"1", "true", "yes", "on"}


def stdout_enabled() -> bool:
    return os.getenv("TASK2_TRACE_STDOUT", "1").lower() not in {"0", "false", "no", "off"}


def _trace_dir() -> Path:
    return Path(os.getenv("TASK2_TRACE_DIR", "/kaggle/working/task2_trace"))


def _max_chars() -> int:
    try:
        return int(os.getenv("TASK2_TRACE_MAX_CHARS", "0"))
    except ValueError:
        return 0


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")
    return safe or "trace"


def _clip(text: str) -> str:
    limit = _max_chars()
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"\n\n[TRACE TRUNCATED at {limit} chars; original_chars={len(text)}]"
    return text


def reset_events() -> None:
    """Start a fresh in-memory trace bucket for one pipeline record."""
    _EVENTS.clear()


def get_events() -> list[dict[str, Any]]:
    """Return a copy of trace events collected for the current pipeline record."""
    return [dict(event) for event in _EVENTS]


def log_text(label: str, text: Any) -> None:
    """Write a trace entry to file, collect it for JSONL output, and optionally stdout."""
    if not enabled():
        return
    raw_body = str(text)
    body = _clip(raw_body)
    safe = _safe_label(label)
    out_dir = _trace_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe}.txt"
    path.write_text(body, encoding="utf-8")
    _EVENTS.append({
        "label": label,
        "safe_label": safe,
        "path": str(path),
        "chars": len(body),
        "original_chars": len(raw_body),
        "truncated": len(body) != len(raw_body),
        "content": body,
    })
    if stdout_enabled():
        print(f"\n===== TASK2 TRACE START: {label} =====", flush=True)
        print(body, flush=True)
        print(f"===== TASK2 TRACE END: {label} =====\n", flush=True)


def log_json(label: str, obj: Any) -> None:
    if not enabled():
        return
    log_text(label, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def compact_example(example: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields that matter for prompt/RAG debugging."""
    execution = example.get("execution") or {}
    verification = example.get("verification") or {}
    return {
        "id": example.get("id"),
        "source_id": example.get("source_id"),
        "detected_problem_type": example.get("detected_problem_type"),
        "answer_type": example.get("answer_type"),
        "question": example.get("question"),
        "module2_extract": example.get("module2_extract"),
        "module3_si_extract": example.get("module3_si_extract"),
        "solution_plan": example.get("solution_plan"),
        "module4_symcode": example.get("module4_symcode"),
        "execution": {
            "status": execution.get("status"),
            "boxed": execution.get("boxed"),
            "numeric_answer": execution.get("numeric_answer"),
            "pred_unit": execution.get("pred_unit"),
        },
        "verification": {
            "status": verification.get("status"),
            "reason": verification.get("reason"),
        },
    }


def compact_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_example(example) for example in examples]
