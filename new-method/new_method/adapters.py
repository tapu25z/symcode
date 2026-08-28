"""Adapters for the existing 7B coder runner and benchmark sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping


LEGACY_7B_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


@dataclass(frozen=True)
class StageTokenBudgets:
    extractor: int = 1400
    ir_repair: int = 1400
    codegen: int = 1800
    code_repair: int = 1800


class Legacy7BCoderAdapter:
    """Expose the old LLMRunner as the text-only callback expected by the pipeline."""

    def __init__(self, runner: Any, budgets: StageTokenBudgets | None = None):
        self.runner = runner
        self.budgets = budgets or StageTokenBudgets()
        self.calls: list[dict[str, Any]] = []

    @property
    def total_generated_tokens(self) -> int:
        return sum(int(item["generated_tokens"]) for item in self.calls)

    def snapshot(self) -> tuple[int, int]:
        return len(self.calls), self.total_generated_tokens

    def calls_since(self, snapshot: tuple[int, int]) -> list[dict[str, Any]]:
        return self.calls[snapshot[0] :]

    def _stage(self, messages: list[dict[str, str]]) -> str:
        system = str(messages[0].get("content", "")) if messages else ""
        if "now repairing a candidate mathematical IR" in system:
            return "ir_repair"
        if "repair a Python program" in system:
            return "code_repair"
        if "normalized JSON payload" in system:
            return "codegen"
        return "extractor"

    def __call__(self, messages: list[dict[str, str]]) -> str:
        stage = self._stage(messages)
        token_budget = int(getattr(self.budgets, stage))
        text, generated_tokens = self.runner.generate_chat(
            messages,
            max_new_tokens_override=token_budget,
            enable_thinking=False,
        )
        self.calls.append({"stage": stage, "generated_tokens": int(generated_tokens), "response": text})
        return str(text)


class LegacySandboxAdapter:
    """Parse the existing sandbox's stdout into the strict structured execution contract."""

    def __init__(self, executor: Callable[..., Mapping[str, Any]], timeout: int = 15):
        self.executor = executor
        self.timeout = int(timeout)

    def __call__(self, code: str) -> dict[str, Any]:
        raw = dict(self.executor(code, mode="symcode", timeout=self.timeout) or {})
        status = raw.get("status")
        stdout = str(raw.get("stdout") or "")
        if status != "success":
            return {"execution_error": raw.get("traceback") or f"sandbox status: {status}", "_sandbox_status": status, "_stdout": stdout, "_traceback": raw.get("traceback")}
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            return {"execution_error": f"expected exactly one JSON output line, got {len(lines)}", "_sandbox_status": status, "_stdout": stdout, "_traceback": None}
        try:
            parsed = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            return {"execution_error": f"invalid JSON output: {exc}", "_sandbox_status": status, "_stdout": stdout, "_traceback": None}
        if not isinstance(parsed, dict):
            return {"execution_error": "JSON output must be an object", "_sandbox_status": status, "_stdout": stdout, "_traceback": None}
        return {**parsed, "_sandbox_status": status, "_stdout": stdout, "_traceback": None}


def build_legacy_7b_runner(
    model_id: str = LEGACY_7B_MODEL_ID,
    load_in_4bit: bool = True,
    max_new_tokens: int = 1800,
    max_input_tokens: int = 6144,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """Instantiate the existing model runner only when a real benchmark explicitly calls this factory."""
    LLMRunner = None
    try:
        from method.model import LLMRunner
    except Exception:
        try:
            from kaggle.method.model import LLMRunner
        except Exception:
            try:
                from method import LLMRunner
            except Exception:
                from kaggle.method import LLMRunner
    if LLMRunner is None:
        raise RuntimeError(
            "legacy LLMRunner dependencies are unavailable. "
            "Please ensure virtual environment is activated (`source venv/bin/activate`) "
            "and dependencies are installed: `pip install torch transformers bitsandbytes accelerate`"
        )
    return LLMRunner(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        temperature=temperature,
        **kwargs,
    )
