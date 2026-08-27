#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight BM25 retrieval for Task 2 few-shot/RAG examples.

This module is dependency-free so it can run on Kaggle without extra installs.
It builds in-memory BM25 indexes from verified SymCode examples and supports
leakage-safe retrieval by excluding the current record id/source_id.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOKEN_RE = re.compile(r"[a-z]+[a-z0-9]*|[0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def tokenize(text: Any) -> List[str]:
    """Tokenize physics text for BM25 matching."""
    s = str(text or "").lower()
    s = (
        s.replace("ω", " omega ")
        .replace("Ω", " ohm ")
        .replace("μ", " u ")
        .replace("µ", " u ")
        .replace("π", " pi ")
        .replace("\\", " ")
    )
    return TOKEN_RE.findall(s)


def normalize_exclude_ids(exclude_ids: Optional[Any]) -> set[str]:
    """Normalize one id or an iterable of ids."""
    if exclude_ids is None:
        return set()
    if isinstance(exclude_ids, (str, int)):
        return {str(exclude_ids).strip()}
    return {str(x).strip() for x in exclude_ids if str(x).strip()}


def example_ids(example: Dict[str, Any]) -> set[str]:
    """Return ids that identify an example's source record."""
    return {
        str(example.get("id", "")).strip(),
        str(example.get("source_id", "")).strip(),
    }


def is_excluded(example: Dict[str, Any], exclude_ids: Optional[Any]) -> bool:
    excluded = normalize_exclude_ids(exclude_ids)
    return bool(excluded and not example_ids(example).isdisjoint(excluded))


def _iter_quantity_text(items: Any) -> Iterable[str]:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("name", "symbol", "value", "unit", "answer_type"):
            val = item.get(key)
            if val not in (None, ""):
                yield str(val)
        options = item.get("options")
        if isinstance(options, dict):
            yield " ".join(f"{k} {v}" for k, v in options.items())


def build_module2_text(example: Dict[str, Any]) -> str:
    """Build retrieval text for extraction examples."""
    mod2 = example.get("module2_extract") or {}
    parts = [
        str(example.get("detected_problem_type", "")),
        str(example.get("answer_type", "")),
        str(example.get("question", "")),
    ]
    parts.extend(_iter_quantity_text(mod2.get("given", [])))
    parts.extend(str(x) for x in mod2.get("conditions", []) if x)
    parts.extend(_iter_quantity_text(mod2.get("target", [])))
    return " ".join(parts)


def build_module4_text(example: Dict[str, Any]) -> str:
    """Build retrieval text for SymCode examples."""
    mod3 = example.get("module3_si_extract") or {}
    plan = example.get("solution_plan") or {}
    code = str(example.get("module4_symcode", ""))
    code_comments = " ".join(
        line.lstrip("# ").strip()
        for line in code.splitlines()
        if line.strip().startswith("#")
    )

    parts = [
        build_module2_text(example),
        str(plan.get("method", "")),
        " ".join(str(x) for x in plan.get("formulas", []) if x),
        " ".join(str(x) for x in plan.get("steps", []) if x),
        code_comments,
    ]
    parts.extend(_iter_quantity_text(mod3.get("given", [])))
    parts.extend(str(x) for x in mod3.get("conditions", []) if x)
    parts.extend(_iter_quantity_text(mod3.get("target", [])))
    return " ".join(parts)


class BM25Index:
    """Small BM25 index over in-memory examples."""

    def __init__(self, examples: List[Dict[str, Any]], text_field: str):
        self.examples = examples
        self.text_field = text_field
        self.term_counts: List[Counter[str]] = []
        self.doc_lens: List[int] = []
        self.doc_freq: Counter[str] = Counter()

        for ex in examples:
            counts = Counter(tokenize(ex.get(text_field, "")))
            self.term_counts.append(counts)
            doc_len = sum(counts.values())
            self.doc_lens.append(doc_len)
            self.doc_freq.update(counts.keys())

        self.n_docs = len(examples)
        self.avg_doc_len = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if not df or not self.n_docs:
            return 0.0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_terms: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        counts = self.term_counts[doc_idx]
        doc_len = self.doc_lens[doc_idx] or 1
        avg_len = self.avg_doc_len or 1.0
        score = 0.0

        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1.0 - b + b * doc_len / avg_len)
            score += self._idf(term) * (tf * (k1 + 1.0)) / denom
        return score

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None,
        exclude_ids: Optional[Any] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        query_terms = tokenize(query)
        if not query_terms or not self.examples:
            return []

        filters = {k: v for k, v in (filters or {}).items() if v and v != "unknown"}
        rows: List[Tuple[Dict[str, Any], float]] = []
        for idx, ex in enumerate(self.examples):
            if is_excluded(ex, exclude_ids):
                continue
            if any(str(ex.get(k, "")) != str(v) for k, v in filters.items()):
                continue
            score = self.score(query_terms, idx)
            if score > 0:
                rows.append((ex, score))

        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:top_k]


def build_rag_indexes(examples: List[Dict[str, Any]]) -> Dict[str, BM25Index]:
    """Attach retrieval text and build Module 2/4 indexes."""
    for ex in examples:
        ex["_rag_m2_text"] = build_module2_text(ex)
        ex["_rag_m4_text"] = build_module4_text(ex)

    return {
        "module2": BM25Index(examples, "_rag_m2_text"),
        "module4": BM25Index(examples, "_rag_m4_text"),
    }


def retrieve_examples(
    bank: Dict[str, Any],
    index_name: str,
    query: str,
    n: int,
    filters: Optional[Dict[str, str]] = None,
    exclude_ids: Optional[Any] = None,
    oversample: int = 6,
) -> List[Dict[str, Any]]:
    """Return top examples from a named RAG index."""
    index = (bank.get("rag") or {}).get(index_name)
    if not index:
        return []
    rows = index.search(
        query=query,
        top_k=max(n * oversample, n),
        filters=filters,
        exclude_ids=exclude_ids,
    )
    return [ex for ex, _score in rows[:n]]


def vote_problem_type_from_rag(
    bank: Dict[str, Any],
    query: str,
    exclude_ids: Optional[Any] = None,
    top_k: int = 8,
    min_top_count: int = 2,
    min_score_ratio: float = 1.10,
) -> Tuple[str, Dict[str, Any]]:
    """Infer problem_type by weighted majority vote from retrieved examples."""
    index = (bank.get("rag") or {}).get("module2")
    if not index:
        return "unknown", {"status": "no_index"}

    rows = index.search(query=query, top_k=top_k, exclude_ids=exclude_ids)
    if not rows:
        return "unknown", {"status": "no_hits"}

    counts: Counter[str] = Counter()
    scores: Dict[str, float] = defaultdict(float)
    retrieved_ids = []
    for ex, score in rows:
        label = str(ex.get("detected_problem_type", "unknown") or "unknown")
        if label == "unknown":
            continue
        counts[label] += 1
        scores[label] += float(score)
        retrieved_ids.append(str(ex.get("id", "")))

    if not scores:
        return "unknown", {"status": "no_labeled_hits", "retrieved_ids": retrieved_ids}

    ranked = sorted(scores.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    ratio = float("inf") if second_score <= 0 else top_score / second_score
    confident = counts[top_label] >= min_top_count and ratio >= min_score_ratio

    info = {
        "status": "pass" if confident else "low_confidence",
        "top_label": top_label,
        "top_count": counts[top_label],
        "score_ratio": ratio,
        "scores": dict(scores),
        "counts": dict(counts),
        "retrieved_ids": retrieved_ids,
    }
    return (top_label if confident else "unknown"), info
