"""
Unified Methods Package:
- Direct (Baseline): Zero-shot Direct Answering
- CoT (Baseline): Chain-of-Thought Natural Language Reasoning
- SymCode (Baseline): Neurosymbolic with SymPy and Independent Mathematical Verifier
- SymPlan (Ours): Verifiable SymPlanner IR Pipeline
"""

from .direct import evaluate_direct
from .cot import evaluate_cot
from .symcode import evaluate_symcode
from .symplan import evaluate_symplan, evaluate_ir_variant, compute_ir_diagnostics

METHOD_EVALUATORS = {
    "Direct": evaluate_direct,
    "CoT": evaluate_cot,
    "SymCode": evaluate_symcode,
    "SymPlan": evaluate_symplan,
    "IR-Full": evaluate_symplan,
    "IR-Codegen": lambda dataset, runner, **kw: evaluate_ir_variant(dataset, runner, variant="IR-Codegen", **kw),
    "IR-BiVerify": lambda dataset, runner, **kw: evaluate_ir_variant(dataset, runner, variant="IR-BiVerify", **kw),
}

__all__ = [
    "evaluate_direct",
    "evaluate_cot",
    "evaluate_symcode",
    "evaluate_symplan",
    "evaluate_ir_variant",
    "compute_ir_diagnostics",
    "METHOD_EVALUATORS",
]
