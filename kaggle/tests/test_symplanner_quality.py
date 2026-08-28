import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kaggle"))

from method.extractor import check_exact_match
from method.prompts import build_symplanner_codegen_messages
from method.problem_hints import build_problem_hints
from method.static_lint import lint_sympy_code
from method.sandbox import execute_code_safely
from method.target_contract import infer_target_spec, parse_planner_contract, format_answer_for_contract
from method.verifier import verify_candidate_answer


class SymPlannerQualityTests(unittest.TestCase):
    def test_common_math_format_variants_are_equivalent(self):
        self.assertTrue(check_exact_match("3*sqrt(13)", r"3\sqrt{13}"))
        self.assertTrue(check_exact_match("11*sqrt(2)", r"11\sqrt2"))
        self.assertTrue(check_exact_match("6 + 9*I", "6+9i"))
        self.assertTrue(check_exact_match("[3, 5, 7]", "3, 5, 7"))
        self.assertTrue(check_exact_match(["3", "5", "7"], "3, 5, 7"))
        self.assertTrue(check_exact_match(["5.00000000000000"], "x=5"))
        self.assertTrue(check_exact_match("x=5", "5"))
        self.assertTrue(check_exact_match(["3.00000000000000", "1.57079632679490"], r"\left( 3, \frac{\pi}{2} \right)"))


    def test_base_notation_is_not_collapsed_to_decimal(self):
        self.assertFalse(check_exact_match("52", "52_8"))
        self.assertTrue(check_exact_match("52_8", "52_8"))
        self.assertEqual(
            format_answer_for_contract(r"Find $6_8\cdot 7_8$. Express your answer in base $8$.", "52", "base_notation"),
            "52_8",
        )


    def test_verifier_fails_closed_for_numeric_guess_and_target_mismatch(self):
        status, _ = verify_candidate_answer("How many students are there?", "17")
        self.assertEqual(status, "unknown")
        status, _ = verify_candidate_answer("Which student has the greatest average speed?", "3.6")
        self.assertEqual(status, "fail")
        status, _ = verify_candidate_answer("Which student has the greatest average speed?", "Evelyn")
        self.assertEqual(status, "unknown")
        status, _ = verify_candidate_answer("How many miles did she travel?", "1.25")
        self.assertEqual(status, "unknown")
        status, _ = verify_candidate_answer("Write the answer in terms of p and q.", "-zeta(3)+pi**2/6")
        self.assertEqual(status, "fail")
        status, _ = verify_candidate_answer("Find all values of x that satisfy the equation.", "5")
        self.assertEqual(status, "unknown")


    def test_planner_contract_recovers_truncated_plan(self):
        note, parsed, errors = parse_planner_contract("```json\n{\"target_unknown\": \"x\"\n", "Which student wins?")
        self.assertTrue(note)
        self.assertTrue(errors)
        self.assertEqual(parsed["answer_type"], "text")
        self.assertEqual(infer_target_spec("Convert the point to polar coordinates")["answer_type"], "tuple")


    def test_codegen_prompt_contains_target_contract(self):
        messages = build_symplanner_codegen_messages("Which student has the greatest speed?", "{}")
        self.assertIn("OUTPUT CONTRACT", messages[-1]["content"])
        self.assertIn('"answer_type": "text"', messages[-1]["content"])


    def test_problem_hints_are_answer_free_but_algorithmic(self):
        hints = build_problem_hints("In total, how many values can be obtained by inserting parentheses?")
        self.assertTrue(any("dynamic programming" in hint for hint in hints))
        hints = build_problem_hints("Solve -4 < 2(x - 1) < 8.")
        self.assertTrue(any("chained inequalities" in hint for hint in hints))


    def test_static_lint_catches_known_sympy_hazards(self):
        findings = lint_sympy_code("sol = sp.solve(eq, x)[0]\nvalue = x.evalf()\nprint('{\"answer\": {}}'.format(value))")
        self.assertGreaterEqual(len(findings), 3)

    def test_symplanner_sandbox_enforces_structured_output(self):
        code = 'import json; print(json.dumps({"answer":"5","canonical_answer":"5","answer_type":"number","unit":None,"variables":{}}))'
        result = execute_code_safely(code, mode="symplanner")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extracted_answer"], "5")
        self.assertEqual(result["answer_type"], "number")
        invalid = execute_code_safely('print("{bad json}")', mode="symplanner")
        self.assertEqual(invalid["status"], "error")

    def test_symplanner_sandbox_accepts_model_json_near_misses(self):
        no_import = 'print(json.dumps({"answer": sp.Rational(14, 3), "canonical_answer": sp.Rational(14, 3), "answer_type": "number", "unit": None}))'
        result = execute_code_safely(no_import, mode="symplanner")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extracted_answer"], "14/3")
        self.assertEqual(result["variables"], {})

        jsonish = r'print("{\"answer\": 14/3, \"canonical_answer\": \"\\frac{14}{3}\", \"answer_type\": \"number\", \"unit\": null}")'
        result = execute_code_safely(jsonish, mode="symplanner")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extracted_answer"], "14/3")

    def test_planner_note_does_not_pollute_numeric_target_inference(self):
        note = '{"strategy":"symbolic equation solving","steps":["Express sin(A) and cos(A) in terms of tan(A)"],"answer_type":"number"}'
        status, _ = verify_candidate_answer("What is tan A?", "2", planner_note=note)
        self.assertEqual(status, "unknown")
        self.assertEqual(infer_target_spec("For which positive real number C does the bound hold?")["answer_type"], "number")
        self.assertEqual(infer_target_spec("What is the value of x+y for the parallelogram coordinates?")["answer_type"], "number")
        self.assertEqual(infer_target_spec("Find the roots of x^2-1.")["answer_type"], "set")
        self.assertEqual(infer_target_spec("What is the smallest n such that all the roots of x^4+x^2+1 are nth roots of unity?")["answer_type"], "number")
        self.assertEqual(infer_target_spec("Enter the ordered triple (p,q,r).")["answer_type"], "tuple")
