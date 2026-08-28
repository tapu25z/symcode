"""
Định nghĩa hệ thống prompt và bộ tạo thông điệp ChatML cho các phương pháp benchmark:
Direct, CoT, SymCode và SymPlanner (Decoupled Divide-and-Plan Pipeline: Planner -> Pure Codegen -> Debug Repair).
Sử dụng phân tích động cây cú pháp Python (AST - Abstract Syntax Tree) tổng quát để chẩn đoán nguyên nhân Timeout không cứng nhắc.
"""

import ast
import re
from typing import Dict, List, Optional, Any

# ==============================================================================
# 1. PLANNER PROMPTS (Turn 1: Phân tích & Lập kế hoạch ngắn gọn, không sinh code)
# ==============================================================================

PLANNER_SYSTEM_PROMPT = r"""You are a careful mathematical planner.

Think through the problem, then output a compact structured plan that will be used by a separate code generator.
The plan must be short, practical, and precise. Do NOT generate Python code.

Output ONLY this JSON object:
{
  "target_unknown": "exact quantity or simplified expression to find",
  "target_type": "number | fraction | tuple | interval | set | entity_name | expression",
  "domain_constraints": "positive integer, real number, 0 <= theta < 360, etc.",
  "given_constants": ["key numbers, parameters and relations from problem"],
  "strategy": "sequential arithmetic OR symbolic equation solving (sp.solve)",
  "steps": ["step 1", "step 2", "step 3"],
  "pitfalls": ["unit/sign/rounding/free-variable pitfalls to avoid"]
}

Rules:
- Read constants from the problem accurately.
- For large summations or systemic relations, prefer analytical identities or pattern recognition.
Keep the final JSON under 150 words."""

# ==============================================================================
# 2. CODEGEN PROMPTS (Turn 2: Sinh mã nguồn Python/SymPy thuần túy 100%)
# ==============================================================================

SYMPLANNER_CODEGEN_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Given one math problem and planner notes, return ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output markdown text outside the code fence. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp, functools, math, fractions if helpful.
2. Define given values and variables clearly with proper domain assumptions (e.g., sp.symbols('x', positive=True, real=True)).
3. Compute the requested target quantity according to target_type and domain_constraints:
   - For sequential word problems: Compute step-by-step using Python/SymPy arithmetic.
   - For algebraic systems: Use sp.symbols(...) with domain assumptions and sp.solve(...).
4. STRICT SYMPY CODE RULES & ANTI-PATTERNS:
   - NEVER use integer division `//` inside SymPy equations or sp.Eq(). Use standard `/` or `sp.Rational(a, b)`.
   - ALWAYS check if a solution list is non-empty before indexing `[0]` (e.g., `ans = sol[0] if sol else None`).
   - For inequalities, use `sp.reduce_inequalities([cond1, cond2], x)` instead of bitwise operators `&`/`|`.
   - For variable substitutions, ALWAYS pass a dictionary: `expr.subs({x: val1, y: val2})`. NEVER pass tuples.
   - ALWAYS filter domain constraints explicitly (e.g., `[s for s in sols if s > 0]`).
   - ANTI-RECURSION RULE: For recursive functions f(n, m), ALWAYS add `@functools.lru_cache(None)` above definition or convert to iterative dynamic programming loop to avoid RecursionError.
   - GEOMETRY & ANGLE RULE: For angle measures, enforce domain constraints (e.g., `angle = angle % 360`).
   - TUPLE PARAMETER RULE: For ordered triples/tuples like (p, q, r), NEVER print symbolic variable names like `(a6, a3, a0)`. Solve for exact numeric values.
   - Never output `None`, `Invalid`, or undefined variables. Do NOT create conditional `if/else` checks that assign `None`.
   - Never call `.evalf()` on Python standard `int` or `float`.
5. FORMATTING & SIMPLIFICATION & LATEX PRINTING:
   - ALWAYS post-process final SymPy expression/number using `sp.nsimplify()`, `sp.radsimp()`, `sp.trigsimp()`, or `sp.expand_complex()` to simplify radicals and convert repeating decimals to exact fractions.
   - Always print the final answer in LaTeX boxed format at the very end.
   - If `final_answer` is a SymPy object or expression, use `sp.latex(final_answer)`:
     `if isinstance(final_answer, (sp.Basic, sp.Matrix)): print(f"\\boxed{{{sp.latex(final_answer)}}}") else: print(f"\\boxed{{{final_answer}}}")`
"""

SYMCODE_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Solve the problem by returning ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp (and math, fractions if helpful).
2. Define all given quantities and formulate equations accurately.
3. Solve for the target quantity symbolically or numerically.
4. Never call `.evalf()` on standard Python int/float.
5. Print ONLY the final answer in LaTeX boxed format at the end:
   print(f"\\boxed{{{final_answer}}}")
"""

# ==============================================================================
# 3. DEBUG / REPAIR PROMPTS (Turn 3: Sửa lỗi mã nguồn có chủ đích)
# ==============================================================================

DEBUG_SYSTEM_PROMPT = r"""You are fixing Python/SymPy code for a math problem based on a detailed diagnosis.

Return ONLY corrected executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

Fix the shown failure:
- Syntax error / Indentation error / Runtime error (AttributeError, TypeError, KeyError, RecursionError)
- Timeout error: Change algorithmic approach! DO NOT output the same code structure. Switch to a lower-complexity strategy.
- RecursionError: Add `@functools.lru_cache(None)` above recursive functions or rewrite using an iterative DP loop.
- Misuse of `//` inside SymPy `sp.Eq()` or passing tuples to `.subs()`
- Unevaluated Python variable name or missing `\boxed{}` output
- Free variables remaining in answer when a concrete value is required
- Verifier diagnosis feedback

Requirements:
1. Ensure the code computes a concrete numerical value, LaTeX-formatted expression, or simplified symbolic result.
2. Print ONLY the final answer in LaTeX boxed format:
   `if isinstance(final_answer, (sp.Basic, sp.Matrix)): print(f"\\boxed{{{sp.latex(final_answer)}}}") else: print(f"\\boxed{{{final_answer}}}")`
"""

# ==============================================================================
# 4. BASELINE PROMPTS (Direct & CoT)
# ==============================================================================

COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""

DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""

SYSTEM_PROMPTS = {
    "Direct": DIRECT_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
    "SymCode": SYMCODE_SYSTEM_PROMPT,
    "SymPlanner": SYMPLANNER_CODEGEN_SYSTEM_PROMPT,
}


# ==============================================================================
# 5. GENERALIZED DYNAMIC AST ANALYSIS & MESSAGE BUILDERS
# ==============================================================================

def remove_thinking_tags(text: str) -> str:
    """Loại bỏ các thẻ <think>...</think> của các mô hình reasoning (Qwen, DeepSeek...)."""
    text = str(text or "")
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[0].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text.strip()


def clean_planner_note(raw_plan: str) -> str:
    """Làm sạch kết quả kế hoạch từ Turn 1."""
    if not raw_plan or not raw_plan.strip():
        return ""
    text = remove_thinking_tags(raw_plan.strip())
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text[:1500].strip()


def extract_code_ast_features(code: str) -> List[str]:
    """
    Phân tích cây cú pháp (AST - Abstract Syntax Tree) tổng quát của đoạn mã ngẫu nhiên
    để tự động trích xuất các thành phần cấu trúc phức tạp mà không rào cứng hàm/số cụ thể.
    """
    features = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # 1. Phát hiện vòng lặp lồng nhau
            if isinstance(node, (ast.While, ast.For)):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, (ast.While, ast.For)):
                        features.append("Nested loop structure (For/While lồng nhau)")
                        break

            # 2. Phân tích các lệnh gọi hàm đại số biểu tượng / solver
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name in ("solve", "solveset", "linsolve", "nonlinsolve", "diophantine", "dsolve", "rsolve"):
                    features.append(f"Symbolic equation solver: `{func_name}()`")
                elif func_name in ("simplify", "expand", "factor", "full_simplify", "nsimplify", "radsimp", "trigsimp"):
                    features.append(f"Heavy symbolic expression transformer: `{func_name}()`")

            # 3. Trích xuất các dải lặp lớn hơn 500
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value > 500:
                        features.append(f"Large iteration range (`range(..., {arg.value})`)")
    except Exception:
        pass

    return list(dict.fromkeys(features))


def diagnose_timeout_cause(code: str) -> str:
    """
    Chẩn đoán Timeout dựa trên cây cú pháp AST tổng quát và quy tắc nguyên tắc chung (Meta-Rules).
    """
    ast_features = extract_code_ast_features(code)
    
    lines = [
        "DIAGNOSIS FOR TIMEOUT FAILURE:",
        "1. Execution exceeded the maximum allowed time limit (15s).",
    ]
    if ast_features:
        lines.append("2. High-complexity structural elements detected via AST analysis:")
        for feat in ast_features:
            lines.append(f"   - {feat}")
            
    lines.extend([
        "3. MANDATORY STRATEGY SWITCH RULE:",
        "   - The previous approach is computationally intractable.",
        "   - You are STRICTLY FORBIDDEN from returning the same code structure.",
        "   - You MUST switch to a fundamentally different strategy (e.g., from symbolic solver to discrete search, or from iterative summation to closed-form analytical representation)."
    ])
    return "\n".join(lines)


def build_planner_messages(question: str) -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 1: Lập kế hoạch phân tích đề bài."""
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n\nThink if needed, then output the compact JSON plan only. Do not write code."}
    ]


def build_symplanner_codegen_messages(question: str, planner_note: str = "") -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 2: Sinh mã nguồn Python/SymPy từ đề bài + Kế hoạch."""
    plan_block = planner_note.strip() or "Planner note unavailable. Solve directly from first principles."
    user_content = f"""# PROBLEM
{question}

# PLANNER NOTES
{plan_block}

Return executable Python code only enclosed in ```python ... ```. Do not write explanations."""
    return [
        {"role": "system", "content": SYMPLANNER_CODEGEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def build_symplanner_debug_messages(
    question: str,
    bad_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None,
    planner_note: str = ""
) -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 3: Sửa lỗi mã nguồn có chủ đích (Pure Code Debug)."""
    feedback_lines = []
    if execution_status != "success":
        feedback_lines.append(f"Execution status: {execution_status.upper()}")
        if execution_status == "timeout":
            timeout_diag = diagnose_timeout_cause(bad_code)
            feedback_lines.append(f"\n{timeout_diag}")
        elif error_tb:
            clean_tb = str(error_tb).strip()
            if len(clean_tb) > 600:
                clean_tb = clean_tb[-600:]
            feedback_lines.append(f"Traceback:\n{clean_tb}")
    else:
        feedback_lines.append("Execution status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        cand_short = str(candidate_answer)[:150]
        feedback_lines.append(f"Candidate answer printed: {cand_short}")
        
    if verification_feedback:
        verif_short = str(verification_feedback)[:600]
        feedback_lines.append(f"Verifier diagnosis: {verif_short}")

    feedback_text = "\n".join(feedback_lines)
    plan_block = planner_note.strip() or "N/A"

    user_text = f"""# PROBLEM
{question}

# PLANNER NOTES
{plan_block}

# PREVIOUS CODE
```python
{str(bad_code).strip()[:1200]}
```

# DIAGNOSIS
{feedback_text}

Fix the issue and return corrected executable Python code only enclosed in ```python ... ```."""

    return [
        {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """Xây dựng thông điệp ChatML chuẩn cho Direct, CoT, SymCode và SymPlanner."""
    if method == "SymPlanner":
        return build_planner_messages(question)
    elif method == "SymCode":
        return [
            {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
            {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM\n\nReturn executable Python code only enclosed in ```python ... ```."}
        ]
    elif method == "CoT":
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"}
        ]
    elif method == "Direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"}
        ]
    else:
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"}
        ]


# Backward compatibility aliases
def build_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    return build_symplanner_debug_messages(
        question=question,
        bad_code=prev_code,
        execution_status=execution_status,
        error_tb=error_tb,
        candidate_answer=candidate_answer,
        verification_status=verification_status,
        verification_feedback=verification_feedback
    )


def build_symplanner_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    return build_symplanner_debug_messages(
        question=question,
        bad_code=prev_code,
        execution_status=execution_status,
        error_tb=error_tb,
        candidate_answer=candidate_answer,
        verification_status=verification_status,
        verification_feedback=verification_feedback
    )
