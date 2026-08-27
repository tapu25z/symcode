#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đọc/ghi dữ liệu, report và file submission cho Kaggle.

Module này chỉ xử lý I/O: đọc input JSON/JSONL, ghi JSONL kết quả,
tổng hợp report, và xuất submission.csv. Logic suy luận nằm ở các module khác.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .executor import extract_boxed
from .verify import extract_unit_from_boxed, parse_numeric_expression, to_submission_unit


# ===== Extracted notebook cell 10 =====
# Handles multiple input formats:
#   - Single-line JSONL (one JSON per line)
#   - Pretty-printed JSON (multi-line per record)
#   - JSON array
#   - Pseudo-JSONL with raw newlines inside string fields

def parse_pseudo_json_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Fallback parser for pseudo-JSONL where fields like "thinking"
    contain raw (unescaped) newlines, breaking standard JSON parsing.

    Uses regex to extract: id, question, thinking, answer, unit.
    """
    blocks = re.split(r"\n\s*\n(?=\{)", text.strip())
    records: List[Dict[str, Any]] = []

    for block in blocks:
        idm = re.search(r'"id"\s*:\s*"([^"]*)"', block)
        qm  = re.search(r'"question"\s*:\s*"(.*?)",\s*"thinking"', block, re.S)
        tm  = re.search(r'"thinking"\s*:\s*"(.*?)",\s*"answer"', block, re.S)
        am  = re.search(r'"answer"\s*:\s*"(.*?)"', block, re.S)
        um  = re.search(r'"unit"\s*:\s*"(.*?)"', block, re.S)

        if not idm or not qm:
            continue

        records.append({
            "id":       idm.group(1),
            "question": qm.group(1).replace('\\"', '"').replace("\n", " ").strip(),
            "thinking": (tm.group(1).replace('\\"', '"').strip() if tm else ""),
            "answer":   (am.group(1).strip() if am else ""),
            "unit":     (um.group(1).strip() if um else ""),
        })

    return records


def read_records(path: str) -> List[Dict[str, Any]]:
    """
    Read records from a JSONL / JSON / pseudo-JSONL file.

    Tries parsing in order:
      1. Entire file as JSON array or single object
      2. Line-by-line JSONL
      3. Pseudo-JSON fallback (for raw newlines in string fields)
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Attempt 1: parse entire file as JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    # Attempt 2: standard JSONL (one valid JSON per line)
    records: List[Dict[str, Any]] = []
    ok_jsonl = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            ok_jsonl = False
            break

    if ok_jsonl and records:
        return records

    # Attempt 3: pseudo-JSONL fallback
    return parse_pseudo_json_blocks(text)


# Các helper ghi file nhỏ, dùng trong runner để lưu pass/fail theo từng sample.
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append a JSON object as a line to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl_objects(path: str) -> List[Dict[str, Any]]:
    """Read valid JSON objects from a JSONL file, skipping malformed lines."""
    p = Path(path)
    if not p.exists():
        return []

    objects: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                objects.append(obj)
    return objects


def reset_file(path: str) -> None:
    """Clear a file's contents."""
    Path(path).write_text("", encoding="utf-8")


# Report gom thống kê để xem nhanh stage nào đang fail nhiều nhất.
def write_report(
    path: str,
    outputs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    total_processed: int,
) -> None:
    """Write a summary report of the pipeline run."""
    pass_by_type   = Counter(o.get("detected_problem_type", "?") for o in outputs)
    fail_by_type   = Counter(f.get("detected_problem_type", "?") for f in failures)
    pass_by_target_answer_type = Counter(o.get("target_answer_type", o.get("answer_type", "?")) for o in outputs)
    fail_by_target_answer_type = Counter(f.get("target_answer_type", f.get("answer_type", "?")) for f in failures)
    fail_by_stage  = Counter(
        f.get("failure_stage", f.get("stage", f.get("execution", {}).get("error_type", "verification")))
        for f in failures
    )
    fail_by_debug_label = Counter(f.get("debug_label", "unknown") for f in failures)

    report = {
        "total_processed":       total_processed,
        "total_pass":            len(outputs),
        "total_fail":            len(failures),
        "total_accounted":       len(outputs) + len(failures),
        "missing_from_accounting": max(total_processed - len(outputs) - len(failures), 0),
        "pass_rate":             (len(outputs) / total_processed) if total_processed else 0,
        "pass_by_type":          dict(pass_by_type),
        "fail_by_type":          dict(fail_by_type),
        "pass_by_target_answer_type": dict(pass_by_target_answer_type),
        "fail_by_target_answer_type": dict(fail_by_target_answer_type),
        "fail_by_stage":         dict(fail_by_stage),
        "fail_by_debug_label":   dict(fail_by_debug_label),
    }

    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Xuất submission theo đúng format Kaggle: id, answer, unit.
_SUBMISSION_UNIT_ALTERNATIVES = (
    r"km/h|m/s\^?2|m/s²|cm/s\^?2|cm/s²|m/s|N/C|V/m|kg/m\^?3|J/m\^?3|"
    r"N\*m\^?2/C\^?2|mWb|uWb|Wb|kohm|Mohm|ohm|Ω|Omega|"
    r"degC|°C|cm|mm|km|kg|mg|g|ms|us|min|Hz|kHz|MHz|GHz|"
    r"mJ|uJ|nJ|kJ|J|mN|uN|kN|N|mV|kV|V|mA|uA|kA|A|"
    r"mF|uF|nF|pF|F|mH|uH|H|mT|uT|T|rad/s|rad|deg|m|s|h|W|%"
)
_SUBMISSION_UNIT_PATTERN = re.compile(
    rf"(?<![A-Za-z])(?:{_SUBMISSION_UNIT_ALTERNATIVES})(?![A-Za-z])"
)


def _strip_balanced_outer_braces(text: str) -> str:
    """Remove one or more pairs of braces that wrap the whole answer."""
    s = text.strip()
    changed = True
    while changed and len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        changed = False
        depth = 0
        wraps = True
        for idx, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and idx != len(s) - 1:
                    wraps = False
                    break
        if wraps:
            s = s[1:-1].strip()
            changed = True
    return s


def _strip_unmatched_edge_braces(text: str) -> str:
    """Remove stray edge braces without touching valid inner LaTeX formulas."""
    s = text.strip()
    while s.endswith("}") and s.count("}") > s.count("{"):
        s = s[:-1].rstrip()
    while s.startswith("{") and s.count("{") > s.count("}"):
        s = s[1:].lstrip()
    return s


def clean_answer_for_submission(boxed: str, pred_unit: str = "") -> str:
    """Tách phần answer khỏi boxed LaTeX; bỏ unit khỏi cột answer."""
    s = str(boxed or "").strip()
    if s.startswith(r"\boxed{"):
        inner = extract_boxed(s)
        if inner is not None:
            s = inner

    submission_unit = to_submission_unit(pred_unit)
    unit_pattern = _SUBMISSION_UNIT_PATTERN
    if submission_unit in {"C", "degC"}:
        unit_pattern = re.compile(
            rf"(?<![A-Za-z])(?:{_SUBMISSION_UNIT_ALTERNATIVES}|C)(?![A-Za-z])"
        )

    cleaned_parts = []
    for part in re.split(r"\s*;\s*", s):
        part = part.replace("\\\\", "\\")
        part = _strip_balanced_outer_braces(part)
        part = part.replace("\\left", "").replace("\\right", "")
        part = part.replace("\\langle", r"\langle").replace("\\rangle", r"\rangle")

        is_numeric_part = parse_numeric_expression(part) is not None or bool(submission_unit)

        if not is_numeric_part:
            part = re.sub(r"(?:\\mu|μ|µ)\s*\\(?:mathrm|text)\s*\{\s*([^}]+?)\s*\}", r"u\1", part)
            part = re.sub(r"\\(?:mathrm|text)\s*\{\s*([^}]+?)\s*\}", r"\1", part)
            part = part.replace(r"\Omega", "ohm")
            part = part.replace(r"\%", "%")
            part = (
                part.replace(r"\,", " ")
                .replace(r"\;", " ")
                .replace(r"\:", " ")
                .replace(r"\!", " ")
                .replace(r"\ ", " ")
            )
            part = re.sub(r"\\quad|\\qquad", " ", part)
            part = _strip_balanced_outer_braces(part)
            part = re.sub(r"\s+", " ", part).strip()
            part = _strip_unmatched_edge_braces(part)
            cleaned_parts.append(part)
            continue

        # Remove units represented as LaTeX commands/wrappers.
        part = re.sub(r"(?:\\mu|μ|µ)\s*\\(?:mathrm|text)\s*\{\s*([^}]+?)\s*\}", "", part)
        part = re.sub(r"\\(?:mathrm|text)\s*\{\s*([^}]+?)\s*\}", "", part)
        part = part.replace(r"\Omega", "")
        part = part.replace(r"\ohm", "")
        part = part.replace(r"\%", "")
        part = part.replace(r"\degree", "")

        # Normalize LaTeX spacing before removing plain trailing unit tokens.
        part = (
            part.replace(r"\,", " ")
            .replace(r"\;", " ")
            .replace(r"\:", " ")
            .replace(r"\!", " ")
            .replace(r"\ ", " ")
        )
        part = re.sub(r"\\quad|\\qquad", " ", part)
        part = unit_pattern.sub("", part)
        part = _strip_balanced_outer_braces(part)
        part = re.sub(r"\\+$", "", part)
        part = re.sub(r"\s+", " ", part).strip()
        part = re.sub(r"\s+([,)>\\\]])", r"\1", part)
        part = re.sub(r"([(<\\\[])\s+", r"\1", part)
        part = _strip_unmatched_edge_braces(part)
        cleaned_parts.append(part)

    return "; ".join(p for p in cleaned_parts if p)


def export_submission_csv(
    records: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    path: str,
) -> None:
    """Xuất file CSV nộp Kaggle theo thứ tự record gốc."""
    by_id = {str(obj.get("id", "")): obj for obj in outputs + failures}
    missing_ids = []

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "unit"])
        writer.writeheader()
        for record in records:
            record_id = str(record.get("id", ""))
            obj = by_id.get(record_id)
            if obj is None:
                missing_ids.append(record_id)
                writer.writerow({"id": record_id, "answer": "", "unit": ""})
                continue

            execution = obj.get("execution", {})
            boxed = execution.get("boxed") or ""
            answer_type = str(obj.get("target_answer_type", obj.get("answer_type", "")))
            pred_unit = ""
            if answer_type not in {"conceptual_qa", "yes_no", "multiple_choice"}:
                pred_unit = extract_unit_from_boxed(boxed)
                if not pred_unit and parse_numeric_expression(boxed) is not None:
                    pred_unit = execution.get("pred_unit", "")
            writer.writerow({
                "id": record_id,
                "answer": clean_answer_for_submission(boxed, pred_unit),
                "unit": to_submission_unit(pred_unit) or "N/A",
            })

    if missing_ids:
        print(f"WARNING: {len(missing_ids)} records missing from submission accounting.", flush=True)
    print(f"Submission CSV written: {path}", flush=True)


def export_submission_json(
    records: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    path: str,
) -> None:
    """Xuất file JSON nộp Kaggle/BTC theo định dạng PredictOutput list."""
    by_id = {str(obj.get("id", "")): obj for obj in outputs + failures}
    missing_ids = []
    
    results = []
    for record in records:
        # Nếu data có query_id, ưu tiên query_id, nếu không thì fallback id
        record_id = str(record.get("query_id", record.get("id", "")))
        obj = by_id.get(record_id)
        
        if obj is None:
            # Fallback nếu object bị xử lý skip
            missing_ids.append(record_id)
            results.append({
                "query_id": record_id,
                "answer": "",
                "unit": "N/A",
                "explanation": "",
                "premises_used": [],
                "reasoning": {"type": "symcode", "code": ""}
            })
            continue
            
        execution = obj.get("execution", {})
        boxed = execution.get("boxed") or ""
        answer_type = str(obj.get("target_answer_type", obj.get("answer_type", "")))
        
        pred_unit = ""
        if answer_type not in {"conceptual_qa", "yes_no", "multiple_choice"}:
            pred_unit = extract_unit_from_boxed(boxed)
            if not pred_unit and parse_numeric_expression(boxed) is not None:
                pred_unit = execution.get("pred_unit", "")
                
        answer_cleaned = clean_answer_for_submission(boxed, pred_unit)
        unit_cleaned = to_submission_unit(pred_unit) or "N/A"
        
        code_symcode = str(obj.get("module4_symcode", ""))

        # Tách tất cả các comment "# ..." trong SymCode làm CoT explanation.
        steps = []
        for line in code_symcode.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                steps.append(line)

        explanation = "\n".join(steps)
        reasoning = {
            "type": "symcode",
            "code": code_symcode.strip()
        }
        
        results.append({
            "query_id": record_id,
            "answer": answer_cleaned,
            "unit": unit_cleaned,
            "explanation": explanation,
            "premises_used": [],
            "reasoning": reasoning
        })
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} records missing from JSON submission accounting.", flush=True)
    print(f"Submission JSON written: {path}", flush=True)
