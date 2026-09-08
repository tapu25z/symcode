"""Four program-aided methods with shared execution, budgets and retry policy."""
import time

from . import prompts
from .common import python_code
from .execution import execute


def solve(question, method, runner, config, executor=execute):
    """No reference answer, dataset label, subject hint, or grading callback here."""
    if method not in prompts.METHODS:
        raise ValueError(method)
    profile = config.get("prompt_profile", "controlled")
    variant = config.get("symplan_variant", "full")
    if profile not in {"controlled", "source"}:
        raise ValueError("Unknown prompt profile")
    if profile == "source" and config["max_retries"] != 0:
        raise ValueError("Source profile requires zero retries")
    if variant not in {"full", "no-extract", "no-plan", "code-only"}:
        raise ValueError("Unknown SymPlan variant")
    start = time.monotonic()
    calls, attempts = [], []
    remaining = config["total_tokens"]

    def generate(stage, system, context="", cap=None):
        nonlocal remaining
        limit = min(remaining, cap or remaining)
        if limit <= 0:
            return ""
        messages = prompts.generation_messages(method, profile, stage, system, question, context)
        result = runner.generate(messages, limit)
        if not 0 <= result["output_tokens"] <= limit:
            raise ValueError("Backend exceeded the token budget")
        remaining -= result["output_tokens"]
        calls.append({"stage": stage, "messages": messages, "max_new_tokens": limit, **result})
        return result["text"]

    prediction, execution_status = None, "not_executed"
    state, plan = "", ""
    if method == "SymPlan" and variant in {"full", "no-plan"}:
        state = generate("representation", prompts.EXTRACT, cap=config["extract_tokens"])
    if method == "SymPlan" and variant in {"full", "no-extract"}:
        plan = generate("planning", prompts.PLAN, "EXTRACTED STATE:\n" + state,
                        cap=config["plan_tokens"])
    context = ("EXTRACTED STATE:\n" + state + "\nPLAN:\n" + plan) if state or plan else ""
    system = prompts.CODE_PROMPTS[method]
    raw = generate("code", system, context, cap=config["code_tokens"])
    for attempt in range(config["max_retries"] + 1):
        code = python_code(raw)
        executed_code = prompts.executable_program(code, method, profile)
        result = executor(executed_code, timeout=config["exec_timeout"],
                          backend=config["executor"], image=config["executor_image"])
        attempts.append({"code": code, "executed_code": executed_code, **result})
        execution_status = result["status"]
        if execution_status == "success":
            prediction = result["answer"]
            break
        if remaining <= 0 or attempt == config["max_retries"]:
            break
        diagnosis = (context + "\nPREVIOUS PROGRAM:\n" + code +
                     "\nEXECUTION STATUS:\n" + result["status"] +
                     "\nEXECUTION DIAGNOSIS:\n" + result["diagnosis"])
        raw = generate("repair", system + prompts.REPAIR, diagnosis, cap=config["code_tokens"])
    return {"prediction": prediction, "execution_status": execution_status,
            "calls": calls, "attempts": attempts,
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "generation_seconds": sum(c["seconds"] for c in calls),
            "solve_seconds": time.monotonic() - start,
            "limit_hits": sum(c["limit_hit"] for c in calls)}
