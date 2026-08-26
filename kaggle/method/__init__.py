"""
Gói module phương pháp benchmark cho các baseline suy luận: Direct, CoT, SymCode, SymReasoner và SymPlanner.
"""

from .prompts import (
    SYSTEM_PROMPTS,
    SYMCODE_SYSTEM_PROMPT,
    SYMREASONER_SYSTEM_PROMPT,
    SYMPLANNER_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    build_prompt_messages,
    build_retry_prompt_messages,
    build_symreasoner_retry_prompt_messages,
    build_symplanner_retry_prompt_messages
)
from .extractor import (
    extract_boxed_content,
    extract_python_code,
    extract_python_code_ast,
    extract_symplanner_code,
    extract_dapper_code,
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
try:
    from .model import LLMRunner
except ImportError:
    LLMRunner = None

from .evaluator import (
    load_dataset_file,
    evaluate_direct_or_cot,
    evaluate_symcode,
    evaluate_symreasoner,
    evaluate_symplanner,
    evaluate_dapper,
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
    "SYMCODE_SYSTEM_PROMPT",
    "SYMREASONER_SYSTEM_PROMPT",
    "SYMPLANNER_SYSTEM_PROMPT",
    "COT_SYSTEM_PROMPT",
    "DIRECT_SYSTEM_PROMPT",
    "build_prompt_messages",
    "build_retry_prompt_messages",
    "build_symreasoner_retry_prompt_messages",
    "build_symplanner_retry_prompt_messages",
    "extract_boxed_content",
    "extract_python_code",
    "extract_python_code_ast",
    "extract_symplanner_code",
    "extract_dapper_code",
    "extract_gsm8k_ground_truth",
    "extract_ground_truth",
    "normalize_answer_str",
    "check_exact_match",
    "execute_code_safely",
    "verify_candidate_answer",
    "LLMRunner",
    "load_dataset_file",
    "evaluate_direct_or_cot",
    "evaluate_symcode",
    "evaluate_symreasoner",
    "evaluate_symplanner",
    "evaluate_dapper",
    "compute_metrics_table",
    "save_benchmark_results",
    "prompts",
    "extractor",
    "sandbox",
    "verifier",
    "evaluator",
    "model"
]



