import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_method.normalizer import build_codegen_payload, is_supported_unit, normalize_problem_ir, validate_normalized_ir
from new_method.problem_ir import normalize_ir_shape, validate_ir
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


def numeric_execution(answer, variables=None):
    return {"answer": answer, "canonical_answer": answer, "answer_type": "number", "unit": None, "variables": variables or {}}


class ContractTests(unittest.TestCase):
    def test_valid_ir_and_answer_pass(self):
        ir = valid_ir()
        self.assertEqual(validate_ir(ir), [])
        normalized = normalize_problem_ir(ir)
        self.assertEqual(validate_normalized_ir(normalized), [])
        self.assertEqual(verify_bidirectional(normalized, numeric_execution(10))["status"], "pass")

    def test_empty_relation_graph_never_passes(self):
        ir = valid_ir()
        ir["relations"] = []
        self.assertTrue(any("at least one relation" in error for error in validate_ir(ir)))
        self.assertEqual(verify_bidirectional(normalize_problem_ir(ir), numeric_execution(999))["status"], "fail")

    def test_missing_metadata_and_unknown_symbol_are_rejected(self):
        ir = valid_ir()
        ir["relations"] = [{"lhs": "p", "rhs": "r*z", "operator": "="}]
        errors = validate_ir(ir)
        self.assertTrue(any("missing fields" in error for error in errors))
        self.assertTrue(any("undeclared symbols" in error for error in errors))

    def test_output_contract_rejects_missing_variables_and_nan(self):
        normalized = normalize_problem_ir(valid_ir())
        missing = verify_bidirectional(normalized, {"answer": 10, "canonical_answer": 10, "answer_type": "number", "unit": None})
        nan = verify_bidirectional(normalized, numeric_execution(math.nan))
        self.assertEqual(missing["status"], "fail")
        self.assertEqual(nan["status"], "fail")

    def test_reverse_check_rejects_wrong_candidate(self):
        result = verify_bidirectional(normalize_problem_ir(valid_ir()), numeric_execution(11))
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any(item["status"] == "fail" for item in result["checks"][0]["reverse"]))

    def test_computational_payload_excludes_prose_metadata(self):
        payload = build_codegen_payload(normalize_problem_ir(valid_ir()))
        self.assertNotIn("source", payload["relations"][0])
        self.assertNotIn("evidence", payload["relations"][0])
        self.assertNotIn("raw", payload["givens"][0])

    def test_math_competition_units_are_supported(self):
        for unit in ("hr", "km/hr", "in", "units", "degrees"):
            self.assertTrue(is_supported_unit(unit), unit)

    def test_missing_unit_fields_are_defaulted_before_validation(self):
        raw = valid_ir()
        del raw["relations"][0]["unit"]
        shaped = normalize_ir_shape(raw)
        self.assertIsNone(shaped["relations"][0]["unit"])
        self.assertEqual(validate_ir(shaped), [])

    def test_symbolic_expression_givens_are_allowed(self):
        ir = valid_ir()
        ir["target_unknown"] = {"name": "value", "symbol": "R", "unit": None, "dimension": "number"}
        ir["givens"] = [{"name": "expression", "symbol": "E", "value": "2*3*4*5+1", "unit": None, "role": "constant", "source": "expression"}]
        ir["relations"] = [{"id": "value", "kind": "definition", "lhs": "R", "rhs": "E", "operator": "=", "unit": None, "source": "expression", "evidence": "expression value", "confidence": 1.0}]
        normalized = normalize_problem_ir(ir)
        self.assertEqual(validate_normalized_ir(normalized), [])
        self.assertEqual(normalized["givens"][0]["quantity"]["status"], "expression")

    def test_integer_condition_accepts_int_idiom(self):
        normalized = normalize_problem_ir(valid_ir())
        normalized["conditions"] = [{"kind": "integer", "expr": "p == int(p)", "source": "integer answer"}]
        result = verify_bidirectional(normalized, numeric_execution(10))
        self.assertEqual(result["status"], "pass", result)

    def test_symbolic_math500_style_answer(self):
        ir = {
            "target_unknown": {"name": "sum", "symbol": "S", "unit": None, "dimension": "symbolic"},
            "givens": [
                {"name": "p", "symbol": "p", "value": "p", "unit": None, "role": "parameter", "source": "named p"},
                {"name": "q", "symbol": "q", "value": "q", "unit": None, "role": "parameter", "source": "named q"},
            ],
            "relations": [{"id": "sum", "kind": "definition", "lhs": "S", "rhs": "p-q", "operator": "=", "unit": None, "source": "reindex", "evidence": "in terms of p and q", "confidence": 1.0}],
            "conditions": [],
            "required_output": {"type": "symbolic", "unit": None, "precision": "exact", "digits": None, "target_count": 1},
        }
        self.assertEqual(validate_ir(ir), [])
        normalized = normalize_problem_ir(ir)
        result = verify_bidirectional(normalized, {"answer": "p - q", "canonical_answer": "p-q", "answer_type": "symbolic", "unit": None, "variables": {}})
        self.assertEqual(result["status"], "pass", result)

    def test_tuple_math500_style_answer(self):
        ir = {
            "target_unknown": {"name": "pair", "symbol": "P", "unit": None, "dimension": "tuple"},
            "givens": [
                {"name": "x", "symbol": "x", "value": 0, "unit": None, "role": "constant", "source": "(0,3)"},
                {"name": "y", "symbol": "y", "value": 3, "unit": None, "role": "constant", "source": "(0,3)"},
            ],
            "relations": [
                {"id": "radius", "kind": "definition", "lhs": "r", "rhs": "sqrt(x**2+y**2)", "operator": "=", "unit": None, "source": "polar", "evidence": "radius", "confidence": 1.0},
                {"id": "angle", "kind": "definition", "lhs": "theta", "rhs": "pi/2", "operator": "=", "unit": None, "source": "polar", "evidence": "angle", "confidence": 1.0},
                {"id": "pair", "kind": "definition", "lhs": "P", "rhs": "Tuple(r,theta)", "operator": "=", "unit": None, "source": "polar", "evidence": "requested pair", "confidence": 1.0},
            ],
            "conditions": [{"kind": "positive", "expr": "r>0", "source": "r>0"}, {"kind": "range", "expr": "theta<2*pi", "source": "theta range"}],
            "required_output": {"type": "tuple", "unit": None, "precision": "exact", "digits": None, "target_count": 1},
        }
        self.assertEqual(validate_ir(ir), [])
        result = verify_bidirectional(normalize_problem_ir(ir), {"answer": "(3, pi/2)", "canonical_answer": "Tuple(3,pi/2)", "answer_type": "tuple", "unit": None, "variables": {"r": 3, "theta": "pi/2"}})
        self.assertEqual(result["status"], "pass", result)

    def test_tuple_answer_accepts_close_numeric_components(self):
        ir = {
            "target_unknown": {"name": "pair", "symbol": "P", "unit": None, "dimension": "tuple"},
            "givens": [
                {"name": "x", "symbol": "x", "value": 0, "unit": None, "role": "constant", "source": "(0,3)"},
                {"name": "y", "symbol": "y", "value": 3, "unit": None, "role": "constant", "source": "(0,3)"},
            ],
            "relations": [
                {"id": "radius", "kind": "definition", "lhs": "r", "rhs": "sqrt(x**2+y**2)", "operator": "=", "unit": None, "source": "polar", "evidence": "radius", "confidence": 1.0},
                {"id": "angle", "kind": "definition", "lhs": "theta", "rhs": "pi/2", "operator": "=", "unit": None, "source": "polar", "evidence": "angle", "confidence": 1.0},
                {"id": "pair", "kind": "definition", "lhs": "P", "rhs": "Tuple(r,theta)", "operator": "=", "unit": None, "source": "polar", "evidence": "requested pair", "confidence": 1.0},
            ],
            "conditions": [],
            "required_output": {"type": "tuple", "unit": None, "precision": "exact", "digits": None, "target_count": 1},
        }
        result = verify_bidirectional(normalize_problem_ir(ir), {"answer": "(3.0, 1.57079632679490)", "canonical_answer": "(3.0, 1.57079632679490)", "answer_type": "tuple", "unit": None, "variables": {"r": 3.0, "theta": 1.57079632679490}})
        self.assertEqual(result["status"], "pass", result)


if __name__ == "__main__":
    unittest.main()
