"""SymPlanner IR: an experimental, verifiable math reasoning pipeline."""

from .problem_ir import ProblemIR, normalize_ir_shape, validate_ir
from .normalizer import normalize_problem_ir, build_codegen_payload
from .relation_verifier import verify_bidirectional
from .config import ABLATIONS, AblationConfig
from .adapters import LEGACY_7B_MODEL_ID, Legacy7BCoderAdapter, LegacySandboxAdapter
from .evaluator import compute_ir_diagnostics, evaluate_ir_variant

__all__ = [
    "ProblemIR",
    "normalize_ir_shape",
    "validate_ir",
    "normalize_problem_ir",
    "build_codegen_payload",
    "verify_bidirectional",
    "ABLATIONS",
    "AblationConfig",
    "LEGACY_7B_MODEL_ID",
    "Legacy7BCoderAdapter",
    "LegacySandboxAdapter",
    "evaluate_ir_variant",
    "compute_ir_diagnostics",
]
