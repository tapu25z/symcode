import json
from pathlib import Path

import pytest

from paperbench.common import atomic_json, boxed, digest, load_records, question_id
from paperbench.engine import solve
from paperbench.execution import execute
from paperbench.grading import grade
from paperbench.prepare import choose_dev
from paperbench.report import summarize
from paperbench.run import load_data, validate_resume
from paperbench.prompts import METHODS, BASELINES, CODE_PROMPTS

CONFIG = {"total_tokens": 30, "extract_tokens": 5, "plan_tokens": 7,
          "code_tokens": 15, "max_retries": 1, "exec_timeout": 2,
          "executor": "local-test", "executor_image": "unused"}


class FakeRunner:
    def __init__(self):
        self.limits = []

    def generate(self, messages, limit):
        self.limits.append(limit)
        return {"text": 'print(r"\\boxed{4}")', "input_tokens": 10,
                "output_tokens": limit, "limit_hit": True, "seconds": 0.01}


def failure(*args, **kwargs):
    return {"status": "error", "answer": None, "stdout": "", "diagnosis": "bad code", "seconds": 0.1}


@pytest.mark.parametrize("method", METHODS)
def test_shared_total_budget_including_repair(method):
    runner = FakeRunner()
    result = solve("2+2", method, runner, CONFIG, executor=failure)
    assert result["output_tokens"] <= 30
    assert sum(runner.limits) == result["output_tokens"]
    assert len(result["attempts"]) == 2
    if method == "SymPlan":
        assert runner.limits == [5, 7, 15, 3]
    assert result["prediction"] is None


def test_success_stops_retries_and_does_not_fallback_to_plan():
    def success(*args, **kwargs):
        return {"status": "success", "answer": "4", "stdout": r"\boxed{4}", "diagnosis": "", "seconds": 0.1}
    runner = FakeRunner()
    result = solve("2+2", "SymPlan", runner, CONFIG, executor=success)
    assert len(result["attempts"]) == 1
    assert result["prediction"] == "4"
    failed = solve("2+2", "SymPlan", FakeRunner(), CONFIG, executor=failure)
    assert failed["prediction"] is None


@pytest.mark.parametrize("method", BASELINES)
def test_baselines_execute_and_keep_their_method_during_repair(method):
    result = solve("2+2", method, FakeRunner(), CONFIG, executor=failure)
    assert [c["stage"] for c in result["calls"]] == ["code", "repair"]
    assert result["calls"][0]["messages"][0]["content"] == CODE_PROMPTS[method]
    assert result["calls"][1]["messages"][0]["content"].startswith(CODE_PROMPTS[method])
    assert len(result["attempts"]) == 2
    assert result["prediction"] is None


@pytest.mark.parametrize("prediction,gold,correct", [
    (r"\frac{2}{4}", r"\frac{1}{2}", True),
    ("(1,2)", "(2,1)", False),
    ("[0,1]", "(0,1)", False),
    ("[0,1)", "[0,1]", False),
    (r"\{1,2\}", r"\{2,1\}", True),
    ("17", "4", False),
    (None, "4", False),
    ("52", "52_8", False),
])
def test_grading_semantics(prediction, gold, correct):
    assert grade(prediction, gold)["correct"] is correct


def test_boxed_extraction_nested_and_no_rationale_fallback():
    assert boxed(r"Earlier 4, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert boxed("I used 42 but did not finish") is None
    assert boxed(r"\boxed{1} \boxed{") is None


def test_real_execution_and_killable_timeout():
    result = execute('import sympy as sp\nprint(r"\\boxed{%s}" % sp.latex(sp.Rational(2,4)))',
                     backend="local-test", timeout=5)
    assert result["status"] == "success"
    assert grade(result["answer"], "1/2")["correct"]
    assert execute("while True: pass", backend="local-test", timeout=0.2)["status"] == "timeout"
    assert execute('print("42")', backend="local-test", timeout=2)["status"] == "unparseable"


def test_resume_refuses_changed_model_or_prompt():
    validate_resume({"model": "a", "hash": "x"}, {"model": "a", "hash": "x"})
    with pytest.raises(ValueError, match="Changed fields"):
        validate_resume({"model": "a", "hash": "x"}, {"model": "b", "hash": "x"})


def test_checkpoint_does_not_silently_accept_duplicates_or_partial_json(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text('{"id":"a","method":"PoT"}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(p)
    p.write_text('{"id":')
    with pytest.raises(json.JSONDecodeError):
        load_records(p)


def test_dev_excludes_test_and_is_reproducible():
    rows = [{"id": str(i), "subject": str(i%2), "level": "1"} for i in range(20)]
    a = choose_dev(rows, {"1", "2"}, 10, 42)
    assert a == choose_dev(rows, {"1", "2"}, 10, 42)
    assert not {r["id"] for r in a} & {"1", "2"}


def test_test_split_rejects_incomplete_data_and_limits(tmp_path):
    rows = [{"id": question_id("q"), "question": "q", "answer": "a"}]
    (tmp_path / "math500_test.jsonl").write_text(json.dumps(rows[0]) + "\n")
    atomic_json(tmp_path / "manifest.json", {"files": {"math500_test.jsonl": {
        "count": 1, "sha256": digest(rows)}}})
    with pytest.raises(ValueError, match="complete benchmark"):
        load_data(tmp_path, "math500", "test", None)


def test_partial_report_uses_common_ids_only():
    def row(id, method, correct):
        return {"id": id, "method": method, "correct": correct,
                "execution_status": "success", "attempts": [{"status": "success"}],
                "output_tokens": 10, "input_tokens": 20, "calls": [1], "solve_seconds": 1,
                "limit_hits": 0, "grading_status": "graded", "level": "1", "subject": "Algebra"}
    report = summarize([row("a", "PoT", False), row("b", "PoT", True),
                        row("a", "SymPlan", True)], ["PoT", "SymPlan"], 2)
    assert report["common_n"] == 1
    assert not report["completed"]
    assert report["methods"]["PoT"]["accuracy"] == 0
    assert report["paired"]["PoT"]["wins"] == 1


class ScriptedRunner:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages, limit):
        return {"text": next(self.responses), "input_tokens": 10,
                "output_tokens": 1, "limit_hit": False, "seconds": 0.01}


@pytest.mark.parametrize("method,function", [("PaL", "solution"), ("PoT", "solver"),
                                            ("SymCode", "compute"), ("SymPlan", "compute")])
def test_program_output_is_the_only_answer_with_real_interpreter(method, function):
    code = f'''import sympy as sp
def {function}():
    x = sp.Symbol("x", positive=True)
    candidates = sp.solve(sp.Eq(3*x, 2), x)
    answer = candidates[0]
    assert 3*answer == 2
    return answer
print(r"\\boxed{{%s}}" % sp.latex({function}()))'''
    responses = [code]
    if method == "SymPlan":
        responses = [r"TARGET: x. Untrusted proposed answer \boxed{999}",
                     "Solve the relation and substitute the root.", code]
    result = solve("Find positive x with 3x=2.", method, ScriptedRunner(responses), CONFIG)
    assert grade(result["prediction"], r"\frac{2}{3}")["correct"]
    assert len(result["attempts"]) == 1
    if method == "SymPlan":
        assert [c["stage"] for c in result["calls"]] == ["representation", "planning", "code"]
        assert "TARGET: x" in result["calls"][1]["messages"][1]["content"]
        assert "Solve the relation" in result["calls"][2]["messages"][1]["content"]
    assert all("Find positive x" in c["messages"][1]["content"] for c in result["calls"])


@pytest.mark.parametrize("method", METHODS)
def test_execution_failure_repairs_but_wrong_executed_answer_does_not(method):
    prefix = ["TARGET: x", "Derive x"] if method == "SymPlan" else []
    result = solve("2+2", method, ScriptedRunner(prefix + [
        'assert False, "relation failed"', 'print(r"\\boxed{4}")']), CONFIG)
    assert [a["status"] for a in result["attempts"]] == ["error", "success"]
    assert result["prediction"] == "4"
    assert "relation failed" in result["calls"][-1]["messages"][1]["content"]
    wrong = solve("2+2", method,
                  ScriptedRunner(prefix + ['print(r"\\boxed{999}")']), CONFIG)
    assert wrong["prediction"] == "999"
    assert len(wrong["attempts"]) == 1


def test_only_four_program_methods_are_registered():
    assert METHODS == ("PaL", "PoT", "SymCode", "SymPlan")
    for method in ("Direct", "CoT", "EoT", "SymPlanner", "SymPlannerExtractOnly"):
        with pytest.raises(ValueError):
            solve("2+2", method, FakeRunner(), CONFIG)


def test_report_compares_all_three_baselines_and_first_attempt_accuracy():
    rows = [{"id": "a", "method": method, "correct": True,
             "first_attempt_correct": method == "SymPlan", "execution_status": "success",
             "attempts": [{"status": "success"}], "output_tokens": 10,
             "input_tokens": 20, "calls": [1], "solve_seconds": 1, "limit_hits": 0,
             "grading_status": "graded", "level": "1", "subject": "Algebra"}
            for method in METHODS]
    report = summarize(rows, METHODS, 1)
    assert report["completed"]
    assert set(report["paired"]) == set(BASELINES)
    assert all("holm_adjusted_p" in p for p in report["paired"].values())
    assert report["methods"]["PaL"]["first_attempt_accuracy"] == 0
    assert report["methods"]["SymPlan"]["first_attempt_accuracy"] == 100
    assert all(r["accuracy"] == 100 for r in report["methods"].values())


@pytest.mark.parametrize("method,function", [("PaL", "solution"), ("PoT", "solver")])
def test_source_profile_executes_returned_function_without_print(method, function):
    config = {**CONFIG, "prompt_profile": "source", "max_retries": 0}
    result = solve("Find half of 3.", method, ScriptedRunner([
        f"from fractions import Fraction\ndef {function}():\n    return Fraction(3, 2)"]), config)
    assert grade(result["prediction"], "3/2")["correct"]
    assert len(result["calls"]) == 1
    assert "print" not in result["attempts"][0]["code"]
    assert "print" in result["attempts"][0]["executed_code"]
    if method == "PaL":
        prompt = result["calls"][0]["messages"][1]["content"]
        assert prompt.count("Q:") == 9
        assert "Find half of 3." in prompt


def test_source_profile_rejects_added_retry():
    with pytest.raises(ValueError, match="zero retries"):
        solve("q", "PaL", FakeRunner(), {**CONFIG, "prompt_profile": "source"})


@pytest.mark.parametrize("variant,stages", [
    ("no-extract", ["planning", "code", "repair"]),
    ("no-plan", ["representation", "code", "repair"]),
    ("code-only", ["code", "repair"]),
])
def test_ablation_removes_only_declared_stage(variant, stages):
    result = solve("2+2", "SymPlan", FakeRunner(),
                   {**CONFIG, "symplan_variant": variant}, executor=failure)
    assert [c["stage"] for c in result["calls"]] == stages
    assert result["output_tokens"] <= CONFIG["total_tokens"]
    assert result["calls"][-2]["messages"][0]["content"] == CODE_PROMPTS["SymPlan"]


def test_vendored_pal_source_matches_pinned_checksums():
    import hashlib
    directory = Path(__file__).resolve().parents[1] / "paperbench/vendor/pal"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert len(provenance["revision"]) == 40
    for name, spec in provenance["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == spec["sha256"]


def test_pipeline_runs_fixed_test_without_a_winner_gate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from paperbench.pipeline import run_suite
    from paperbench.run import source_hash
    settings = {"model": "qwen3-4b", "executor": "docker", "suites": ["controlled"],
                "source_hash": source_hash()}
    atomic_json(tmp_path / "suite.json", {"settings": settings, "revision": "a"*40})
    commands = []
    monkeypatch.setattr("paperbench.pipeline.subprocess.run", lambda cmd, **kw: commands.append(cmd))
    run_suite(SimpleNamespace(output=tmp_path, **{k: settings[k] for k in ["model", "executor", "suites"]}))
    runs = [c for c in commands if "paperbench.run" in c]
    assert [c[c.index("--phase")+1] for c in runs] == ["smoke", "dev", "dev", "test", "test"]
    assert all(c[c.index("--revision")+1] == "a"*40 for c in runs)
    assert commands[-1][2] == "paperbench.export"
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"


def test_protocol_locks_runtime_and_model_identity():
    from paperbench.run import PROTOCOL_KEYS
    assert {"model_id", "model_revision", "versions", "python", "cuda", "gpu",
            "prompt_profile", "symplan_variant"} <= set(PROTOCOL_KEYS)


def test_export_rejects_partial_and_exports_complete_heldout_tables(tmp_path):
    from paperbench.export import export
    from paperbench.run import PROTOCOL_KEYS, source_hash
    atomic_json(tmp_path / "suite.json", {"settings": {"suites": ["controlled"], "source_hash": source_hash()},
                                         "revision": "test-revision"})
    for dataset, n in [("math500", 500), ("gsm8k", 1319)]:
        directory = tmp_path / "controlled" / f"{dataset}_test"
        config = {k: "test" for k in PROTOCOL_KEYS}
        config.update(methods=list(METHODS), model_id="test-model", model_revision="test-revision",
                      source_hash=source_hash(), phase="test", n=n, dataset=dataset,
                      prompt_profile="controlled", symplan_variant="full")
        atomic_json(directory / "manifest.json", {"config": config})
        atomic_json(tmp_path / "controlled/protocol.lock.json", {"protocol": {k: config[k] for k in PROTOCOL_KEYS}})
        rows = [{"id": str(i), "method": m, "correct": m == "SymPlan", "first_attempt_correct": False,
                 "grading_status": "graded", "execution_status": "success", "attempts": [{"status": "success"}],
                 "output_tokens": 20, "input_tokens": 30, "calls": [1], "solve_seconds": 1,
                 "limit_hits": 0, "level": "1", "subject": "Algebra"}
                for i in range(n) for m in METHODS]
        (directory / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    output = export(tmp_path)
    assert "SymPlan" in (output / "results.tex").read_text()
    assert (output / "metrics.csv").read_text().count("test-model") == 8
    records_path = tmp_path / "controlled/math500_test/records.jsonl"
    records_path.write_text(records_path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        export(tmp_path)
