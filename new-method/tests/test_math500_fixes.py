import unittest
from new_method.scoring import check_math500_equivalence
from new_method.normalizer import normalize_problem_ir, normalize_expression
from new_method.pipeline import SymPlannerIRPipeline, RUNTIME_HEADER
from new_method.relation_verifier import verify_bidirectional


def dummy_legacy_match(pred, gt):
    return str(pred).strip() == str(gt).strip()


def dummy_legacy_normalize(s):
    return str(s).strip()


class TestUniversalMathIR(unittest.TestCase):

    def test_scoring_universal_math_cases(self):
        cases = [
            # 1. Integers & Negative integers
            ("-125.0", -125, "-125", True),
            ("78.0", 78, "78", True),
            ("2000.0", 2000, "2000", True),
            ("12.0", 12, "12", True),
            ("83.0", 83, "83", True),
            ("5", 5, "5", True),
            
            # 2. Fractions
            ("0.38880000000062864", "243/625", r"\frac{243}{625}", True),
            ("4.666666666666667", "14/3", r"\frac{14}{3}", True),
            ("3/2", "3/2", r"\frac{3}{2}", True),
            ("1.25", 1.25, "1.25", True),
            
            # 3. Radicals & Algebraic numbers
            ("15.556349186104047", "11*sqrt(2)", r"11\sqrt{2}", True),
            ("98.99494936611667", "70*sqrt(2)", r"70\sqrt{2}", True),
            ("10.816653826391969", "3*sqrt(13)", r"3\sqrt{13}", True),
            
            # 4. Complex Numbers
            ("6.0 + 9.0*I", "6 + 9*I", "6+9i", True),
            ("6 + -5*I", "6 - 5*I", "6 - 5i", True),
            ("-2 + 7*I", "-2 + 7*I", "-2 + 7i", True),
            
            # 5. Polar & Coordinates Tuples
            ("(3.00000000000000, 1.57079632679490)", "(3, pi/2)", r"\left( 3, \frac{\pi}{2} \right)", True),
            
            # 6. Number Theory & Base Suffixes
            ("52", "52", "52_8", True),
            ("52_8", "52_8", "52_8", True),
            
            # 7. Intermediate Algebra large constants
            ("13535", 13535, "13535", True),
        ]
        for pred, canon, gold, exp in cases:
            res = check_math500_equivalence(pred, canon, gold, dummy_legacy_match, dummy_legacy_normalize)
            self.assertEqual(res, exp, f"Failed for pred={pred}, canon={canon}, gold={gold}")

    def test_normalizer_preserves_constants_across_domains(self):
        self.assertEqual(normalize_expression(r"45\pi"), "45*pi")
        self.assertEqual(normalize_expression("2 - 3*I"), "2 - 3*I")
        self.assertEqual(normalize_expression("sqrt(242)"), "sqrt(242)")
        self.assertEqual(normalize_expression("comb(10, 4)"), "comb(10, 4)")
        self.assertEqual(normalize_expression("n % 7"), "n % 7")

    def test_universal_pipeline_no_false_invalid_ir(self):
        pipeline = SymPlannerIRPipeline(lambda x: "{}", lambda x: {})
        
        # Test Number Theory IR
        ir_nt = {
            "target_unknown": {"name": "remainder", "symbol": "r"},
            "givens": [{"name": "n", "symbol": "n", "value": 2003}],
            "relations": [{"id": "r1", "lhs": "r", "rhs": "n % 7", "operator": "="}],
            "required_output": {"type": "number"},
        }
        self.assertEqual(pipeline._fatal_ir_errors(ir_nt, []), [])

        # Test Combinatorics IR
        ir_comb = {
            "target_unknown": {"name": "ways", "symbol": "W"},
            "givens": [{"name": "n", "symbol": "n", "value": 10}, {"name": "k", "symbol": "k", "value": 4}],
            "relations": [{"id": "r1", "lhs": "W", "rhs": "comb(n, k)", "operator": "="}],
            "required_output": {"type": "number"},
        }
        self.assertEqual(pipeline._fatal_ir_errors(ir_comb, []), [])

    def test_fail_safe_verifier_across_operators(self):
        # 1. Modulo relation
        ir_mod = {
            "target_unknown": {"name": "remainder", "symbol": "r"},
            "givens": [{"name": "n", "symbol": "n", "value": 2, "quantity": {"canonical_value": 2}}],
            "relations": [{"id": "r1", "lhs": "n", "rhs": "7", "operator": "%"}],
            "conditions": [],
            "required_output": {"type": "number"},
        }
        exec_mod = {"answer": 2, "canonical_answer": 2, "answer_type": "number", "unit": None, "variables": {"n": 2, "r": 2}}
        res_mod = verify_bidirectional(ir_mod, exec_mod)
        self.assertIn(res_mod["status"], ["pass", "fail", "unknown"])

        # 2. Custom geometry relation
        ir_geom = {
            "target_unknown": {"name": "dist", "symbol": "d"},
            "givens": [{"name": "p1", "symbol": "p1", "value": "(0, 0)"}, {"name": "p2", "symbol": "p2", "value": "(3, 4)"}],
            "relations": [{"id": "r1", "lhs": "d", "rhs": "sqrt((3-0)**2 + (4-0)**2)", "operator": "="}],
            "conditions": [],
            "required_output": {"type": "number"},
        }
        exec_geom = {"answer": 5, "canonical_answer": 5, "answer_type": "number", "unit": None, "variables": {"d": 5}}
        res_geom = verify_bidirectional(ir_geom, exec_geom)
        self.assertEqual(res_geom["status"], "pass")


if __name__ == "__main__":
    unittest.main()
