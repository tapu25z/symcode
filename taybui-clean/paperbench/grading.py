"""One evaluator for every method. Gold is never provided to generation/repair."""
import re


def base_literal(text):
    text = text.strip().replace(" ", "").replace("$", "")
    match = re.fullmatch(r"([0-9][0-9A-Fa-f]*)_\{?(\d+)\}?", text)
    return (match[1].upper().lstrip("0") or "0", int(match[2])) if match else None


def grade(prediction, gold):
    from math_verify import LatexExtractionConfig, parse, verify
    if prediction is None:
        return {"correct": False, "status": "missing_prediction"}
    # Math-Verify 0.8 can drop a numeric base subscript. Preserve that semantic
    # distinction symmetrically for all methods before invoking the parser.
    pred_base, gold_base = base_literal(prediction), base_literal(gold)
    if pred_base is not None or gold_base is not None:
        return {"correct": pred_base == gold_base, "status": "graded_base_notation"}
    try:
        # Wrapping the already-extracted answer prevents matching an intermediate
        # number in the model's rationale. String fallback is exact only.
        expected = parse(r"\boxed{" + gold + "}", extraction_config=[LatexExtractionConfig()],
                         parsing_timeout=3)
        actual = parse(r"\boxed{" + prediction + "}", extraction_config=[LatexExtractionConfig()],
                       parsing_timeout=3)
        correct = bool(verify(expected, actual, timeout_seconds=3, strict=True))
        return {"correct": correct, "status": "graded" if expected and actual else "parse_failure"}
    except Exception as exc:
        return {"correct": False, "status": "grader_error", "error": str(exc)}
