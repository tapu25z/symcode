import unittest
from new_method.scoring import check_math500_equivalence
from new_method.normalizer import normalize_problem_ir, normalize_expression
from new_method.pipeline import SymPlannerIRPipeline, RUNTIME_HEADER
from new_method.relation_verifier import verify_bidirectional


def dummy_legacy_match(pred, gt):
    return str(pred).strip() == str(gt).strip()


def dummy_legacy_normalize(s):
    return str(s).strip()


class TestMath500Fixes(unittest.TestCase):

    def test_scoring_edge_cases(self):
        cases = [
            ("-125.0", -125, "-125", True),
            ("0.38880000000062864", "243/625", r"\frac{243}{625}", True),
            ("1.25", 1.25, "1.25", True),
            ("6.0 + 9.0*I", "6 + 9*I", "6+9i", True),
            ("15.556349186104047", "11*sqrt(2)", r"11\sqrt{2}", True),
            ("98.99494936611667", "70*sqrt(2)", r"70\sqrt{2}", True),
            ("(3.00000000000000, 1.57079632679490)", "(3, pi/2)", r"\left( 3, \frac{\pi}{2} \right)", True),
            ("4.666666666666667", "14/3", r"\frac{14}{3}", True),
            ("13535", 13535, "13535", True),
            ("78.0", 78, "78", True),
            ("2000.0", 2000, "2000", True),
            ("12.0", 12, "12", True),
            ("83.0", 83, "83", True),
            ("3/2", "3/2", r"\frac{3}{2}", True),
            ("5", 5, "5", True),
            ("52", "52", "52_8", True),
            ("52_8", "52_8", "52_8", True),
        ]
        for pred, canon, gold, exp in cases:
            res = check_math500_equivalence(pred, canon, gold, dummy_legacy_match, dummy_legacy_normalize)
            self.assertEqual(res, exp, f"Failed for pred={pred}, canon={canon}, gold={gold}")

    def test_normalizer_preserves_constants(self):
        self.assertEqual(normalize_expression(r"45\pi"), "45*pi")
        self.assertEqual(normalize_expression("2 - 3*I"), "2 - 3*I")
        self.assertEqual(normalize_expression("sqrt(242)"), "sqrt(242)")

    def test_fatal_ir_errors_relaxation(self):
        pipeline = SymPlannerIRPipeline(lambda x: "{}", lambda x: {})
        ir_valid = {
            "target_unknown": {"name": "ans", "symbol": "x"},
            "givens": [{"name": "a", "symbol": "a", "value": 5}],
            "relations": [],
            "required_output": {"type": "number"},
        }
        fatal = pipeline._fatal_ir_errors(ir_valid, [])
        self.assertEqual(fatal, [])

    def test_verifier_handles_modulo_and_unknown_operators_without_crash(self):
        ir = {
            "target_unknown": {"name": "remainder", "symbol": "r"},
            "givens": [{"name": "n", "symbol": "n", "value": 2, "quantity": {"canonical_value": 2}}],
            "relations": [
                {"id": "r1", "lhs": "n", "rhs": "7", "operator": "%"}
            ],
            "conditions": [],
            "required_output": {"type": "number"},
        }
        execution = {
            "answer": 3,
            "canonical_answer": 3,
            "answer_type": "number",
            "unit": None,
            "variables": {"n": 2, "r": 3}
        }
        # Must not raise KeyError: '%'
        res = verify_bidirectional(ir, execution)
        self.assertIn(res["status"], ["pass", "fail", "unknown"])


if __name__ == "__main__":
    unittest.main()
