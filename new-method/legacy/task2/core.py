#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility layer cho Kaggle pipeline.

`core.py` giờ chỉ giữ runtime config và re-export API cũ để notebook/script hiện tại
vẫn chạy được. Logic thật đã được chia sang các file nhỏ hơn:
io, fewshot, prompts, llm, module2, module3_units, module4_codegen, executor,
verify, runner.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Cấu hình mặc định. `run_inference.py` có thể override trực tiếp các biến này.
_XAI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LORA_PATH = os.getenv("HF_LORA_PATH", "")
DEFAULT_MODULE2_LORA_PATH = os.getenv("HF_MODULE2_LORA_PATH", "")
DEFAULT_HOST = ""
DEFAULT_BNB_COMPUTE_DTYPE = os.getenv("HF_BNB_COMPUTE_DTYPE", "auto")
DEFAULT_MODEL_DTYPE = os.getenv("HF_MODEL_DTYPE", "auto")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("HF_TOP_P", "1.0"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("HF_REPETITION_PENALTY", "1.1"))
DEFAULT_MAX_INPUT_TOKENS = int(os.getenv("HF_MAX_INPUT_TOKENS", "4096"))
DEFAULT_TOKENIZER_PADDING_SIDE = os.getenv("HF_TOKENIZER_PADDING_SIDE", "right")
DEFAULT_RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "bm25").lower()
DEFAULT_BGE_MODEL = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
DEFAULT_BGE_INDEX_PATH = os.getenv(
    "BGE_INDEX_PATH",
    str(_XAI_ROOT / "retrieval_index" / "bge_faiss.index"),
)
DEFAULT_BGE_METADATA_PATH = os.getenv(
    "BGE_METADATA_PATH",
    str(_XAI_ROOT / "retrieval_index" / "bge_metadata.jsonl"),
)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "1.0"))
EXEC_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "8"))
NUM_FEW_SHOT = int(os.getenv("NUM_FEW_SHOT", "2"))
ENABLE_DEBUG = os.getenv("ENABLE_DEBUG", "1") not in {"0", "false", "False"}

# Runtime state được set trong run_inference.py trước khi gọi runner.
MODEL_NAME = DEFAULT_MODEL
LORA_PATH = DEFAULT_LORA_PATH
MODULE2_LORA_PATH = DEFAULT_MODULE2_LORA_PATH
HOST = DEFAULT_HOST
API_KEY: Optional[str] = None
LOAD_IN_4BIT = os.getenv("HF_LOAD_IN_4BIT", "0") in {"1", "true", "True"}
LOAD_IN_8BIT = os.getenv("HF_LOAD_IN_8BIT", "0") in {"1", "true", "True"}
BNB_COMPUTE_DTYPE = DEFAULT_BNB_COMPUTE_DTYPE
MODEL_DTYPE = DEFAULT_MODEL_DTYPE
MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
TEMPERATURE = DEFAULT_TEMPERATURE
TOP_P = DEFAULT_TOP_P
REPETITION_PENALTY = DEFAULT_REPETITION_PENALTY
MAX_INPUT_TOKENS = DEFAULT_MAX_INPUT_TOKENS
TOKENIZER_PADDING_SIDE = DEFAULT_TOKENIZER_PADDING_SIDE
RETRIEVAL_BACKEND = DEFAULT_RETRIEVAL_BACKEND
BGE_MODEL_NAME = DEFAULT_BGE_MODEL
BGE_INDEX_PATH = DEFAULT_BGE_INDEX_PATH
BGE_METADATA_PATH = DEFAULT_BGE_METADATA_PATH
client: Optional[Any] = None
bank: Dict[str, Any] = {"all": [], "by_problem_type": {}, "by_answer_type": {}}
bge_retriever: Optional[Any] = None

# Re-export API cũ để code đang `from kaggle_pipeline import core` không bị gãy.
from .executor import execute_code, extract_boxed, is_clean_boxed_stdout, quick_code_safety_check
from .embedding_retriever import BGEEmbeddingRetriever, build_document_text, build_query_text
from .fewshot import load_few_shot_bank, select_few_shot_for_module2, select_few_shot_for_module4
from .io import append_jsonl, clean_answer_for_submission, export_submission_csv, export_submission_json, parse_pseudo_json_blocks, read_jsonl_objects, read_records, reset_file, write_report
from .knowledge_guides import select_knowledge_guides
from .llm import build_client, chat_messages
from .module2 import detect_answer_type, extract_json_block, generate_module2, generate_problem_type, has_explicit_mc_options, infer_problem_type, normalize_question_text, validate_module2
from .module3_units import canonicalize_unit_text, clean_unit, normalize_quantity, normalize_to_si, normalize_unit, normalize_value_unit_pair, parse_numeric_value, process_module3
from .module4_codegen import build_module4_payload, build_module4_payload_from_example, clean_direct_answer, debug_module4, extract_python_code, generate_and_execute_with_debug, generate_direct_conceptual_answer, generate_module4, has_step_comments
from .prompts import ANSWER_TYPE_SYSTEMS, CONCEPTUAL_SYSTEM, DEBUG_PROMPT, MCQ_SYSTEM, MODULE_2_SYSTEM, NUMERIC_SYSTEM, PROBLEM_TYPE_SYSTEM, YESNO_SYSTEM
from .runner import build_failure_output_obj, build_output_obj, classify_pipeline_result, run_pipeline_records
from .verify import choose_prediction_value_for_gold, convert_value_to_si, extract_number_unit_pairs, extract_unit_from_boxed, is_text_answer_type, latex_to_float, normalize_latex_unit, normalize_text_answer, to_submission_unit, units_equivalent, verify_against_gold
