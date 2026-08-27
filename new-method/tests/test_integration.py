import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle"))

from new_method.adapters import Legacy7BCoderAdapter, LegacySandboxAdapter, StageTokenBudgets
from new_method.evaluator import evaluate_ir_variant
from new_method.pipeline import SymPlannerIRPipeline
from new_method.prompts import codegen_prompt, extraction_prompt
from new_method.scoring import check_math500_equivalence
from method import check_exact_match, normalize_answer_str


VALID_IR = {
    "target_unknown": {"name": "answer", "symbol": "x", "unit": None, "dimension": "number"},
    "givens": [{"name": "value", "symbol": "a", "value": 10, "unit": None, "role": "constant", "source": "10"}],
    "relations": [{"id": "answer", "kind": "definition", "lhs": "x", "rhs": "a", "operator": "=", "unit": None, "source": "same value", "evidence": "answer is the value", "confidence": 1.0}],
    "conditions": [],
    "required_output": {"type": "number", "unit": None, "precision": "exact", "digits": None, "target_count": 1},
}


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0), 7


def structured_execution(answer=10):
    return {"answer": answer, "canonical_answer": answer, "answer_type": "number", "unit": None, "variables": {}}


class IntegrationTests(unittest.TestCase):
    def test_legacy_model_adapter_uses_stage_budget_without_loading_model(self):
        runner = FakeRunner(["ir", "code"])
        adapter = Legacy7BCoderAdapter(runner, StageTokenBudgets(extractor=101, codegen=202))
        adapter(extraction_prompt("x"))
        adapter(codegen_prompt({"x": 1}))
        self.assertEqual(runner.calls[0]["max_new_tokens_override"], 101)
        self.assertEqual(runner.calls[1]["max_new_tokens_override"], 202)
        self.assertEqual(adapter.total_generated_tokens, 14)

    def test_sandbox_adapter_parses_exactly_one_json_line(self):
        payload = structured_execution()
        adapter = LegacySandboxAdapter(lambda code, **kwargs: {"status": "success", "stdout": json.dumps(payload) + "\n", "traceback": None})
        result = adapter("print('ignored')")
        self.assertEqual(result["answer"], 10)
        self.assertEqual(result["_sandbox_status"], "success")

    def test_ablation_switch_separates_codegen_from_semantic_verifier(self):
        responses = [json.dumps(VALID_IR), "print('code')"]
        codegen_only = SymPlannerIRPipeline(lambda messages: responses.pop(0), lambda code: structured_execution(11), ablation="IR-Codegen")
        self.assertEqual(codegen_only.run("answer is 10")["status"], "pass")
        responses = [json.dumps(VALID_IR), "print('code')"]
        verified = SymPlannerIRPipeline(lambda messages: responses.pop(0), lambda code: structured_execution(11), ablation="IR-BiVerify")
        self.assertEqual(verified.run("answer is 10")["status"], "fail")

    def test_evaluator_emits_legacy_compatible_result_and_checkpoint(self):
        runner = FakeRunner([json.dumps(VALID_IR), "print('code')"])
        sandbox = lambda code, **kwargs: {"status": "success", "stdout": json.dumps(structured_execution()) + "\n", "traceback": None}
        dataset = [{"question": "answer is 10", "answer": "10", "raw": {"answer": "10"}, "subject": "Algebra", "level": 1, "level_label": "Level 1"}]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "result.json")
            results = evaluate_ir_variant(
                dataset, runner, sandbox, variant="IR-Full", checkpoint_file=checkpoint,
                ground_truth_fn=lambda raw: str(raw["answer"]),
                match_fn=lambda pred, gold: str(pred) == str(gold),
            )
            self.assertEqual(results[0]["verification_status"], "pass")
            self.assertTrue(results[0]["is_correct"])
            self.assertEqual(results[0]["generated_tokens"], 14)
            self.assertTrue(Path(checkpoint).exists())

    def test_math500_matrix_and_infinity_scoring(self):
        matrix_gold = r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}"
        self.assertTrue(check_math500_equivalence("Matrix([[-1,0],[0,-1]])", "Matrix([[-1,0],[0,-1]])", matrix_gold, check_exact_match, normalize_answer_str))
        self.assertTrue(check_math500_equivalence("(2, oo)", "Interval.open(2,oo)", r"(2,\infty)", check_exact_match, normalize_answer_str))


if __name__ == "__main__":
    unittest.main()
