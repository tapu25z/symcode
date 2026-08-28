"""SymPlanner IR: an experimental, verifiable math reasoning pipeline."""

from .problem_ir import (
    ProblemIR,
    acg_to_legacy_ir,
    empty_acg_ir,
    legacy_to_acg_ir,
    normalize_acg_shape,
    normalize_ir_shape,
    validate_acg_ir,
    validate_ir,
)
from .normalizer import build_codegen_payload, classify_unit, normalize_problem_ir
from .relation_verifier import verify_bidirectional
from .solver_planner import plan_solver
from .direct_solver import try_direct_solve
from .config import ABLATIONS, AblationConfig
from .adapters import LEGACY_7B_MODEL_ID, Legacy7BCoderAdapter, LegacySandboxAdapter
from .evaluator import compute_ir_diagnostics, evaluate_ir_variant

__all__ = [
    "ProblemIR",
    "empty_acg_ir",
    "normalize_acg_shape",
    "validate_acg_ir",
    "legacy_to_acg_ir",
    "acg_to_legacy_ir",
    "normalize_ir_shape",
    "validate_ir",
    "normalize_problem_ir",
    "build_codegen_payload",
    "classify_unit",
    "verify_bidirectional",
    "plan_solver",
    "try_direct_solve",
    "ABLATIONS",
    "AblationConfig",
    "LEGACY_7B_MODEL_ID",
    "Legacy7BCoderAdapter",
    "LegacySandboxAdapter",
    "evaluate_ir_variant",
    "compute_ir_diagnostics",
]
