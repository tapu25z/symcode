import contextlib
import io
import json
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_method.normalizer import normalize_quantity
from new_method.pipeline import RUNTIME_HEADER, SymPlannerIRPipeline
from new_method.problem_ir import acg_to_legacy_ir, legacy_to_acg_ir, normalize_acg_shape, validate_acg_ir
from new_method.scoring import check_math500_equivalence
from new_method.solver_planner import plan_solver


def _acg_json():
    return {
        "version": "acg-ir-v1",
        "problem_metadata": {"dataset": None, "domain_hints": ["narrative"]},
        "target": {"id": "x", "name": "answer", "symbol": "x", "unit": None, "dimension": "number", "output_type": "number", "precision": "exact", "target_count": 1},
        "nodes": [{"id": "a", "symbol": "a", "name": "given", "node_type": "quantity", "value": 10, "raw_value": 10, "unit": "stickers", "role": "given", "source": "10 stickers", "evidence": "10 stickers", "confidence": 1.0}],
        "edges": [{"id": "answer", "kind": "definition", "intent": "copy", "operation": "evaluate", "lhs": "x", "rhs": "a", "operator": "=", "inputs": ["a"], "outputs": ["x"], "unit": None, "tags": ["opaque_unit"], "source": "answer is given", "evidence": "answer is 10", "confidence": 1.0, "executable": True}],
        "conditions": [], "solver_hints": [], "extraction_notes": [],
    }


class ACGTests(unittest.TestCase):
    def test_unknown_units_are_preserved_without_failure(self):
        quantity = normalize_quantity("10 stickers")
        self.assertEqual(quantity["status"], "ok")
        self.assertEqual(quantity["unit"], "stickers")
        self.assertEqual(quantity["unit_class"], "opaque")

    def test_acg_shape_validation_and_legacy_adapter(self):
        graph = normalize_acg_shape(_acg_json())
        self.assertEqual(validate_acg_ir(graph), [])
        legacy = acg_to_legacy_ir(graph)
        self.assertEqual(legacy["target_unknown"]["symbol"], "x")
        self.assertEqual(legacy["relations"][0]["lhs"], "x")
        roundtrip = legacy_to_acg_ir(legacy)
        self.assertEqual(roundtrip["version"], "acg-ir-v1")
        self.assertEqual(roundtrip["edges"][0]["lhs"], "x")

    def test_planner_uses_graph_structure_not_dataset_label(self):
        graph = normalize_acg_shape(_acg_json())
        self.assertEqual(plan_solver(graph)["strategy"], "sequential_eval")
        graph["edges"].append({
            "id": "bound", "kind": "constraint", "intent": "domain", "operation": "solve",
            "lhs": "x", "rhs": "20", "operator": "<", "source": "bound", "evidence": "bound",
        })
        self.assertIn(plan_solver(graph)["strategy"], {"symbolic_solve", "hybrid"})

    def test_acg_pipeline_keeps_plan_and_uses_existing_execution_contract(self):
        responses = [json.dumps(_acg_json()), "print(json.dumps({'answer': 10, 'canonical_answer': 10, 'answer_type': 'number', 'unit': None, 'variables': {'x': 10}}))"]

        def execute(code):
            return {"answer": 10, "canonical_answer": 10, "answer_type": "number", "unit": None, "variables": {"x": 10}}

        pipeline = SymPlannerIRPipeline(lambda messages: responses.pop(0), execute, max_repairs=0, ablation="ACG")
        result = pipeline.run("The answer is 10 stickers.")
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["variant"], "ACG")
        self.assertEqual(result["solver_plan"]["strategy"], "sequential_eval")
        self.assertEqual(result["payload"]["representation"], "acg-ir-v1")

    def test_acg_constraint_graph_handles_mixed_narrative_algebra(self):
        graph = _acg_json()
        graph["target"] = {**graph["target"], "name": "x", "symbol": "x"}
        graph["nodes"] = [
            {"id": "denali_dogs", "symbol": "denali_dogs", "name": "Denali dogs", "node_type": "quantity", "value": 16, "raw_value": 16, "unit": "dogs", "role": "given", "source": "16 dogs", "evidence": "16 dogs", "confidence": 1.0},
            {"id": "nate_dogs", "symbol": "nate_dogs", "name": "Nate dogs", "node_type": "quantity", "value": 12, "raw_value": 12, "unit": "dogs", "role": "given", "source": "12 dogs", "evidence": "12 dogs", "confidence": 1.0},
        ]
        graph["edges"] = [{
            "id": "ratio_equal", "kind": "constraint", "intent": "ratio_equivalence", "operation": "solve",
            "lhs": "(16 + 4*x)/12", "rhs": "(16 + x)/(12 - x)", "operator": "=",
            "inputs": ["x"], "outputs": ["x"], "unit": None, "tags": ["ratio", "dogs"],
            "source": "two pay ratios are equal", "evidence": "ratio ... would be the same", "confidence": 1.0, "executable": True,
        }]
        graph["conditions"] = [{"id": "nonzero", "kind": "domain", "expr": "x != 0", "symbols": ["x"], "source": "x != 0", "confidence": 1.0}]
        responses = [json.dumps(graph), """x = sp.symbols('x')
solutions = sp.solve(sp.Eq((16 + 4*x)/12, (16 + x)/(12 - x)), x)
answer = [candidate for candidate in solutions if candidate != 0][0]
print(json.dumps({'answer': enc(answer), 'canonical_answer': enc(answer), 'answer_type': 'number', 'unit': None, 'variables': {'x': enc(answer)}}))"""]
        def execute(code):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(code, {})
            return json.loads(stream.getvalue().strip())

        pipeline = SymPlannerIRPipeline(lambda messages: responses.pop(0), execute, max_repairs=0, ablation="ACG")
        result = pipeline.run("Denali and Nate have 16 and 12 dogs; solve the ratio condition for x.")
        self.assertEqual(result["solver_plan"]["strategy"], "symbolic_solve")
        self.assertEqual(result["status"], "pass", result)

    def test_tuple_serialization_and_scalar_scoring_do_not_emit_sympy_mul_warning(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(RUNTIME_HEADER + "\nprint(json.dumps({'answer': enc(sp.Tuple(-114 + 2*sp.sqrt(3777))), 'canonical_answer': enc(sp.Tuple(-114 + 2*sp.sqrt(3777))), 'answer_type': 'tuple', 'unit': None, 'variables': {}}))", {})
        parsed = json.loads(stream.getvalue())
        self.assertEqual(parsed["answer_type"], "tuple")
        self.assertIsInstance(parsed["answer"], list)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matched = check_math500_equivalence(
                "(-114 + 2*sqrt(3777),)",
                "(-114 + 2*sqrt(3777),)",
                "5",
                lambda left, right: str(left) == str(right),
                lambda value: str(value).strip(),
            )
        self.assertFalse(matched)
        self.assertFalse(any("non-Expr arguments in Mul" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
