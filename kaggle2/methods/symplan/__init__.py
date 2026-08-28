"""
SymPlan Method Package (Verifiable SymPlanner IR Pipeline - Ours)
"""

from .problem_ir import ProblemIR, normalize_ir_shape, validate_ir, extract_json_object
from .normalizer import normalize_problem_ir, build_codegen_payload
from .relation_verifier import verify_bidirectional, validate_execution_output
from .config import ABLATIONS, AblationConfig
from .adapters import LEGACY_7B_MODEL_ID, Legacy7BCoderAdapter, LegacySandboxAdapter, StageTokenBudgets, build_legacy_7b_runner
from .pipeline import SymPlannerIRPipeline
from .scoring import check_math500_equivalence
from .evaluator import compute_ir_diagnostics, evaluate_ir_variant, evaluate_ir_full, evaluate_symplan

__all__ = [
    "ProblemIR",
    "normalize_ir_shape",
    "validate_ir",
    "extract_json_object",
    "normalize_problem_ir",
    "build_codegen_payload",
    "verify_bidirectional",
    "validate_execution_output",
    "ABLATIONS",
    "AblationConfig",
    "LEGACY_7B_MODEL_ID",
    "Legacy7BCoderAdapter",
    "LegacySandboxAdapter",
    "StageTokenBudgets",
    "build_legacy_7b_runner",
    "SymPlannerIRPipeline",
    "check_math500_equivalence",
    "evaluate_ir_variant",
    "evaluate_ir_full",
    "evaluate_symplan",
    "compute_ir_diagnostics",
]
