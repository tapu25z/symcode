import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method.extractor import check_exact_match
from method.prompts import build_extract_messages, build_planner_messages, build_symplanner_codegen_messages, build_symplanner_debug_messages, format_problem_hints
from method.problem_hints import build_problem_hints
from method.static_lint import lint_sympy_code
from method.sandbox import execute_code_safely
from method.target_contract import infer_target_spec, parse_planner_contract, format_answer_for_contract
from method.verifier import verify_candidate_answer
from method.evaluator import evaluate_symplanner
from method.direct import build_messages as build_direct_messages
from method.cot import build_messages as build_cot_messages
from method.symcode import build_messages as build_symcode_messages
from method.symplanner import build_extract_messages as build_symplanner_folder_extract_messages, build_codegen_messages as build_symplanner_folder_codegen_messages


class FakeAblationLLM:
    def __init__(self):
        self.calls = []

    def generate_chat(self, messages, **kwargs):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "Write the plan only" in prompt:
            return "1. Add the two numbers.\n2. Print the sum.", 5
        if "Return executable Python code" in prompt:
            return '```python\nprint("\\\\boxed{4}")\n```', 7
        return "# Target: the sum\n# Given: 2 and 2\n# Constraints: none", 3


class SymPlannerQualityTests(unittest.TestCase):
    def test_method_folder_imports_are_available(self):
        self.assertIn("Solve the following math problem directly", build_direct_messages("1+1")[0]["content"])
        self.assertIn("step-by-step", build_cot_messages("1+1")[0]["content"])
        self.assertIn("executable Python code", build_symcode_messages("1+1")[0]["content"])
        self.assertIn("extract the mathematical state", build_symplanner_folder_extract_messages("1+1")[0]["content"])
        self.assertIn("OUTPUT REQUIREMENT", build_symplanner_folder_codegen_messages("1+1", "{}")[-1]["content"])

    def test_common_math_format_variants_are_equivalent(self):
        self.assertTrue(check_exact_match("3*sqrt(13)", r"3\sqrt{13}"))
        self.assertTrue(check_exact_match("11*sqrt(2)", r"11\sqrt2"))
        self.assertTrue(check_exact_match("6 + 9*I", "6+9i"))
        self.assertTrue(check_exact_match("[3, 5, 7]", "3, 5, 7"))
        self.assertTrue(check_exact_match(["3", "5", "7"], "3, 5, 7"))
        self.assertTrue(check_exact_match(["5.00000000000000"], "x=5"))
        self.assertTrue(check_exact_match("x=5", "5"))
        self.assertTrue(check_exact_match(["3.00000000000000", "1.57079632679490"], r"\left( 3, \frac{\pi}{2} \right)"))
        self.assertTrue(check_exact_match("4/3", r"\frac43"))
        self.assertTrue(check_exact_match("1/2", r"\frac12"))
        self.assertTrue(check_exact_match("1,450,000", "1450000"))

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
        status, _ = verify_candidate_answer(r"Write the answer in terms of $p$ and $q$.", "Sum(lerchphi(1, 3, j + 1), (j, 1, oo))")
        self.assertEqual(status, "fail")
        status, _ = verify_candidate_answer("Find all values of x that satisfy the equation.", "5")
        self.assertEqual(status, "unknown")


    def test_planner_contract_recovers_truncated_plan(self):
        note, parsed, errors = parse_planner_contract("```json\n{\"target_unknown\": \"x\"\n", "Which student wins?")
        self.assertTrue(note)
        self.assertTrue(errors)
        self.assertEqual(parsed["answer_type"], "text")
        self.assertEqual(infer_target_spec("Convert the point to polar coordinates")["answer_type"], "tuple")


    def test_codegen_prompt_contains_compact_output_requirement(self):
        messages = build_symplanner_codegen_messages("Which student has the greatest speed?", "{}")
        self.assertIn("OUTPUT REQUIREMENT", messages[-1]["content"])
        self.assertIn("Answer type: text", messages[-1]["content"])
        self.assertNotIn('"answer_type": "text"', messages[-1]["content"])

    def test_simple_symplanner_prompt_chain(self):
        extract_messages = build_extract_messages("Triangle problem")
        self.assertIn("Extract the mathematical state", extract_messages[-1]["content"])
        messages = build_planner_messages("Triangle problem", "# Target: area")
        self.assertEqual(len(messages), 2)
        self.assertIn("# EXTRACTED STATE", messages[-1]["content"])
        self.assertIn("# Target: area", messages[-1]["content"])

    def test_symplanner_ablation_turn_counts(self):
        dataset = [{"question": "What is 2 + 2?", "answer": "#### 4", "subject": "Arithmetic"}]

        cases = [
            ("extract_only", 2, True, False),
            ("plan_only", 2, False, True),
            ("none", 1, False, False),
        ]
        for ablation, expected_calls, use_extract, use_plan in cases:
            with self.subTest(ablation=ablation):
                llm = FakeAblationLLM()
                results = evaluate_symplanner(
                    dataset,
                    llm,
                    max_retries=0,
                    verbose=False,
                    ablation=ablation,
                    method_name=f"Test{ablation}",
                )
                self.assertEqual(len(llm.calls), expected_calls)
                self.assertTrue(results[0]["is_correct"])
                self.assertEqual(results[0]["symplanner_ablation"], ablation)
                self.assertEqual(results[0]["symplanner_use_extract"], use_extract)
                self.assertEqual(results[0]["symplanner_use_plan"], use_plan)
                self.assertEqual(bool(results[0]["extraction_note"]), use_extract)
                self.assertEqual(bool(results[0]["planner_note"]), use_plan)

    def test_planner_accepts_compact_labeled_format(self):
        raw = """# Subject: geometry
# Target: the fastest student
# Given: each student's distance and time
# Step 1: compute each average speed
# Step 2: compare the speeds
# Step 3: choose the greatest speed
# Answer type: text"""
        note, parsed, errors = parse_planner_contract(raw, "Which student has the greatest speed?")
        self.assertTrue(note)
        self.assertFalse(errors)
        self.assertEqual(parsed.get("subject"), "geometry")
        self.assertEqual(parsed["target_unknown"], "the fastest student")
        self.assertEqual(parsed["answer_type"], "text")
        self.assertEqual(len(parsed["steps"]), 3)

    def test_planner_accepts_numbered_format_used_by_prompt(self):
        raw = """1. Identify the target quotient.
2. Divide the polynomial by the divisor.
3. Print the quotient polynomial."""
        note, parsed, errors = parse_planner_contract(raw, "Find the quotient when x^2 - 1 is divided by x + 1.")
        self.assertTrue(note)
        self.assertFalse(errors)
        self.assertEqual(parsed["answer_type"], "symbolic")
        self.assertEqual(len(parsed["steps"]), 3)

    def test_planner_accepts_structured_state_labels(self):
        raw = """# Subject: number_theory
# Target: number of valid x
# Variables: x is a positive integer; n is a divisor count
# Relations: x^2 divides 10!
# Constraints: x > 0; integer exponents only
# Step 1: factor 10!
# Step 2: bound each prime exponent in x
# Answer type: number"""
        note, parsed, errors = parse_planner_contract(raw, "How many positive x have x^2 as a factor of 10!?")
        self.assertTrue(note)
        self.assertFalse(errors)
        self.assertIn("x is a positive integer", parsed["variables"])
        self.assertIn("x^2 divides 10!", parsed["relations"])
        self.assertIn("x > 0", parsed["constraints"])


    def test_debug_prompt_uses_compact_output_requirement(self):
        messages = build_symplanner_debug_messages(
            "What is 2 + 2?",
            "print(2 + )",
            execution_status="error",
            error_tb="SyntaxError",
        )
        self.assertIn("OUTPUT REQUIREMENT", messages[-1]["content"])
        self.assertNotIn('"answer_type":', messages[-1]["content"])

    def test_problem_hints_are_not_injected_into_simple_symplanner(self):
        question = "In total, how many values can be obtained by inserting parentheses?"
        self.assertIn("dynamic programming", format_problem_hints(question))
        messages = build_symplanner_codegen_messages(question, "{}", subject="Counting & Probability")
        self.assertNotIn("PROBLEM-SPECIFIC ALGORITHM HINTS", messages[-1]["content"])
        debug_messages = build_symplanner_debug_messages(question, "print(121)", subject="Counting & Probability")
        self.assertNotIn("PROBLEM-SPECIFIC ALGORITHM HINTS", debug_messages[-1]["content"])


    def test_problem_hints_are_answer_free_but_algorithmic(self):
        hints = build_problem_hints("In total, how many values can be obtained by inserting parentheses?")
        self.assertTrue(any("dynamic programming" in hint for hint in hints))
        hints = build_problem_hints("Solve -4 < 2(x - 1) < 8.")
        self.assertTrue(any("chained inequalities" in hint for hint in hints))
        hints = build_problem_hints("Find a double sum in terms of p and q.")
        self.assertTrue(any("group terms" in hint for hint in hints))
        self.assertFalse(any("p - q" in hint for hint in hints))


    def test_static_lint_catches_known_sympy_hazards(self):
        findings = lint_sympy_code("sol = sp.solve(eq, x)[0]\nvalue = x.evalf()\nprint('{\"answer\": {}}'.format(value))")
        self.assertGreaterEqual(len(findings), 3)

    def test_static_lint_catches_recursion_hazard(self):
        findings = lint_sympy_code("def eval_expr(sub_expr):\n    return eval_expr(sub_expr[:-1])")
        self.assertTrue(any("recursive function call detected" in f for f in findings))
        findings = lint_sympy_code("while True:\n    pass")
        self.assertTrue(any("unbounded while True" in f for f in findings))

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
        self.assertEqual(infer_target_spec("Find the quotient when x^6 - 3 is divided by x + 1.")["answer_type"], "symbolic")
        self.assertEqual(infer_target_spec(r"Find the vector $\\mathbf{v}$ such that a dot v = 2.")["answer_type"], "matrix")

    def test_verifier_flags_detectable_wrong_strategies_for_retry(self):
        status, feedback = verify_candidate_answer(
            "In how many ways can 7 people sit around a round table if no two of 3 people sit next to each other?",
            "576",
            "total_permutations = sp.factorial(6)\nrestricted_permutations = sp.factorial(4)*sp.factorial(3)\nanswer = total_permutations - restricted_permutations",
        )
        self.assertEqual(status, "fail")
        self.assertIn("no-adjacency", feedback)

        status, _ = verify_candidate_answer(
            "Find the smallest C for which ||A v|| <= C ||v|| for all two-dimensional vectors v. The matrix is [[2,3],[0,-2]].",
            "2",
            "A = sp.Matrix([[2,3],[0,-2]])\nC = max(abs(e) for e in A.eigenvals())",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            "You have seven bags of gold coins. What is the smallest number of coins?",
            "x",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            "What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?",
            "729",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            r"Suppose $\sin D=0.7$. What is $DE$? [asy] pair D,E,F; F=(0,0); D=(sqrt(51),7); E=(0,7); [/asy]",
            "4.9",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            "A worker makes three end-of-year deposits with compound interest. What rate is needed?",
            "49.03",
            "equation = sp.Eq(A, P * (1 + r)**n)",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            "Find the quotient when x^6 - 3 is divided by x + 1.",
            "-6",
            "quotient, remainder = sp.div(x**6 - 3, x + 1)\nprint(remainder)",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            r"Remmy wants to divide $10$ by $\\frac{2}{3}$. By what number should he multiply $10$ to get the answer?",
            "15",
        )
        self.assertEqual(status, "fail")

        status, _ = verify_candidate_answer(
            "The polynomial x^3 - 3x^2 + 4x - 1 is a factor of x^9 + px^6 + qx^3 + r. Enter the ordered triple (p,q,r).",
            "(1, 0, 0)",
            "quotient, remainder = sp.div(g, f)\nprint((1, 0, 0))",
        )
        self.assertEqual(status, "fail")
