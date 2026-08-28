"""
Gói module phương pháp benchmark cho 4 phương pháp suy luận: Direct, CoT, SymCode và SymPlanner.
"""

from .prompts import (
    SYSTEM_PROMPTS,
    DIRECT_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    SYMCODE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYMPLANNER_CODEGEN_SYSTEM_PROMPT,
    DEBUG_SYSTEM_PROMPT,
    build_prompt_messages,
    build_planner_messages,
    clean_planner_note,
    build_symplanner_codegen_messages,
    build_symplanner_debug_messages,
    build_retry_prompt_messages,
    build_symplanner_retry_prompt_messages
)
from .extractor import (
    extract_boxed_content,
    extract_answer_fallback,
    extract_python_code,
    extract_symplanner_code,
    extract_gsm8k_ground_truth,
    extract_ground_truth,
    normalize_answer_str,
    check_exact_match
)
from .sandbox import (
    execute_code_safely
)
from .verifier import (
    verify_candidate_answer
)
from .target_contract import infer_target_spec, parse_planner_contract
from .static_lint import lint_sympy_code
try:
    from .model import LLMRunner
except ImportError:
    LLMRunner = None

from .evaluator import (
    load_dataset_file,
    evaluate_direct_or_cot,
    evaluate_symcode,
    evaluate_symplanner,
    compute_metrics_table,
    save_benchmark_results
)

from . import prompts
from . import extractor
from . import sandbox
from . import verifier
from . import evaluator
try:
    from . import model
except ImportError:
    pass

__all__ = [
    "SYSTEM_PROMPTS",
    "DIRECT_SYSTEM_PROMPT",
    "COT_SYSTEM_PROMPT",
    "SYMCODE_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "SYMPLANNER_CODEGEN_SYSTEM_PROMPT",
    "DEBUG_SYSTEM_PROMPT",
    "build_prompt_messages",
    "build_planner_messages",
    "clean_planner_note",
    "build_symplanner_codegen_messages",
    "build_symplanner_debug_messages",
    "build_retry_prompt_messages",
    "build_symplanner_retry_prompt_messages",
    "extract_boxed_content",
    "extract_answer_fallback",
    "extract_python_code",
    "extract_symplanner_code",
    "extract_gsm8k_ground_truth",
    "extract_ground_truth",
    "normalize_answer_str",
    "check_exact_match",
    "execute_code_safely",
    "verify_candidate_answer",
    "infer_target_spec",
    "parse_planner_contract",
    "lint_sympy_code",
    "LLMRunner",
    "load_dataset_file",
    "evaluate_direct_or_cot",
    "evaluate_symcode",
    "evaluate_symplanner",
    "compute_metrics_table",
    "save_benchmark_results",
    "prompts",
    "extractor",
    "sandbox",
    "verifier",
    "evaluator",
    "model"
]



