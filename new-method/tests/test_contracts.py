import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_method.normalizer import build_codegen_payload, normalize_problem_ir, validate_normalized_ir
from new_method.problem_ir import validate_ir
from new_method.relation_verifier import verify_bidirectional


def valid_ir():
    return {
        "target_unknown": {"name": "part", "symbol": "p", "unit": None, "dimension": "number"},
        "givens": [
            {"name": "rate", "symbol": "r", "value": "20%", "unit": "%", "role": "constant", "source": "20%"},
            {"name": "whole", "symbol": "w", "value": 50, "unit": None, "role": "constant", "source": "50"},
        ],
        "relations": [
            {"id": "percent", "kind": "proportion", "lhs": "p", "rhs": "r*w", "operator": "=", "unit": None, "source": "wording", "evidence": "20% of 50", "confidence": 1.0}
        ],
        "conditions": [],
        "required_output": {"type": "number", "unit": None, "precision": "exact", "digits": None, "target_count": 1},
    }


class ContractTests(unittest.TestCase):
    def test_valid_ir_and_answer_pass(self):
        ir = valid_ir()
        self.assertEqual(validate_ir(ir), [])
        normalized = normalize_problem_ir(ir)
        self.assertEqual(validate_normalized_ir(normalized), [])
        self.assertEqual(verify_bidirectional(normalized, {"answer": 10, "unit": None, "variables": {}})["status"], "pass")

    def test_empty_relation_graph_never_passes(self):
        ir = valid_ir()
        ir["relations"] = []
        self.assertTrue(any("at least one relation" in error for error in validate_ir(ir)))
        self.assertEqual(verify_bidirectional(normalize_problem_ir(ir), {"answer": 999, "unit": None, "variables": {}})["status"], "fail")

    def test_missing_metadata_and_unknown_symbol_are_rejected(self):
        ir = valid_ir()
        ir["relations"] = [{"lhs": "p", "rhs": "r*z", "operator": "="}]
        errors = validate_ir(ir)
        self.assertTrue(any("missing fields" in error for error in errors))
        self.assertTrue(any("undeclared symbols" in error for error in errors))

    def test_output_contract_rejects_missing_variables_and_nan(self):
        normalized = normalize_problem_ir(valid_ir())
        missing = verify_bidirectional(normalized, {"answer": 10, "unit": None})
        nan = verify_bidirectional(normalized, {"answer": math.nan, "unit": None, "variables": {}})
        self.assertEqual(missing["status"], "fail")
        self.assertEqual(nan["status"], "fail")

    def test_reverse_check_rejects_wrong_candidate(self):
        result = verify_bidirectional(normalize_problem_ir(valid_ir()), {"answer": 11, "unit": None, "variables": {}})
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any(item["status"] == "fail" for item in result["checks"][0]["reverse"]))

    def test_computational_payload_excludes_prose_metadata(self):
        payload = build_codegen_payload(normalize_problem_ir(valid_ir()))
        self.assertNotIn("source", payload["relations"][0])
        self.assertNotIn("evidence", payload["relations"][0])
        self.assertNotIn("raw", payload["givens"][0])


if __name__ == "__main__":
    unittest.main()
