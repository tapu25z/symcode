"""
Định nghĩa hệ thống prompt và bộ tạo thông điệp ChatML cho các phương pháp benchmark:
Direct, CoT và SymCode (Neurosymbolic Equation Solving với SymPy và vòng lặp Verifier).
"""

from typing import Dict, List, Optional

# Prompt cho phương pháp SymCode: Lập kế hoạch CoT + Mã nguồn SymPy thực thi
SYMCODE_SYSTEM_PROMPT = """You are an expert mathematical reasoner and symbolic computation specialist.

To solve the problem accurately without missing any variables, constraints, or boundary conditions, follow this two-step process:

### Step 1: Concise Mathematical Breakdown (Chain-of-Thought)
- Briefly explain your step-by-step mathematical reasoning.
- Explicitly identify all given quantities, target unknowns, and define each variable with its domain assumptions (e.g., positive integer, real number).
- Set up the governing algebraic equations and relationships clearly.

### Step 2: Executable SymPy Python Script
- Provide the complete, self-contained Python script enclosed in a single ```python ... ``` block.
- Import SymPy as `import sympy as sp`.
- Define variables using `sp.symbols(...)` with appropriate assumptions (e.g. `positive=True, integer=True`).
- Formulate and solve the equations using `sp.Eq(...)` and `sp.solve(...)`.
- Verify domain constraints and filter out extraneous roots.
- At the end of the script, print ONLY the final answer in LaTeX boxed format:
  print(f"\\\\boxed{{{final_answer}}}")"""

# Prompt cho phương pháp SymReasoner: Constraint-Grounded Synthesis (CGS) với cơ chế lọc nghiệm an toàn
SYMREASONER_SYSTEM_PROMPT = """You are an expert mathematical reasoner and symbolic computation specialist applying Constraint-Grounded Synthesis (CGS).

To solve the problem accurately and avoid extraneous solutions or invalid domains, follow this structured process:

### Step 1: Concise Mathematical Breakdown & Constraint Extraction
- Briefly explain your step-by-step mathematical reasoning.
- Explicitly identify given quantities, target unknowns, and domain constraints (e.g. positive real numbers for geometry/dimensions, integers for counting/divisors, valid probabilities in [0, 1], non-zero denominators).
- Set up the governing algebraic equations or direct mathematical formulations clearly.

### Step 2: Executable Python Script with Robust Constraint Grounding
- Provide the complete, self-contained Python script enclosed in a single ```python ... ``` block.
- Import SymPy as `import sympy as sp`.
- Define variables using `sp.symbols(...)` with appropriate domain assumptions (e.g. `positive=True, integer=True, real=True`).
- Formulate and solve the problem using symbolic solving (`sp.solve(...)`) or direct algebraic/arithmetic computation.
- If multiple candidate solutions/roots are returned by `sp.solve(...)`:
  - Define a lightweight domain filter `def check_constraints(candidate): -> bool` (e.g. checking `candidate > 0` or non-zero denominator).
  - Filter the candidates: `valid_sols = [s for s in solutions if check_constraints(s)]`.
  - Pick the valid solution, or fallback to the primary candidate if none pass: `final_answer = valid_sols[0] if valid_sols else solutions[0]`.
- For symbolic identities (e.g. expressing an expression in terms of given symbols like p, q), directly construct and simplify the symbolic result.
- CRITICAL RULES:
  - NEVER output `None`, `null`, or `"Invalid"`. Always output a concrete computed value or simplified symbolic expression.
  - Wrap constraint checks inside `try...except` to prevent runtime crashes.
  - At the end of the script, print ONLY the final answer in LaTeX boxed format:
    print(f"\\\\boxed{{{final_answer}}}")"""

# Prompt cho phương pháp SymPlanner: Divide-and-Plan Neurosymbolic Program Synthesis (Divide -> Plan -> SymCode Execution -> Guarded Repair)
SYMPLANNER_SYSTEM_PROMPT = """You are an expert mathematical reasoner and symbolic computation specialist applying the Divide-and-Plan Neurosymbolic framework.

To solve the problem with high precision without missing variables or boundary conditions, you MUST strictly follow three structured stages:

### Stage 1: DIVIDE (Problem State & Constraint Extraction)
Deconstruct the problem into explicit state components:
- Target Unknown: The exact quantity, value, or simplified algebraic expression to compute.
- Given Quantities: Known constants, given parameters, and relationships.
- Domain Constraints: Explicit mathematical and physical boundaries (e.g., positive real numbers, integer constraints, non-zero denominators).

### Stage 2: PLAN (Algorithmic Solution Strategy)
Outline the step-by-step symbolic derivation procedure:
1. Define the system of algebraic/symbolic equations.
2. Specify the exact solving strategy (e.g., substitution, matrix reduction, `sp.solve`, or direct algebraic simplification).
3. Plan how to validate candidate solutions against the Stage 1 domain constraints to eliminate extraneous roots.

### Stage 3: EXECUTE (Guarded SymCode Generation)
Translate your plan into a complete, self-contained Python script enclosed in a single ```python ... ``` block:
1. Import SymPy as `import sympy as sp`.
2. Define variables with appropriate assumptions (e.g., `sp.symbols('x', positive=True, real=True)`).
3. Formulate and solve the equations symbolically using `sp.solve(...)` or direct computation.
4. If multiple roots exist, filter them using Stage 1 constraints:
   `valid_sols = [s for s in solutions if check_condition(s)]`
   `final_answer = valid_sols[0] if valid_sols else solutions[0]`
5. CRITICAL RULE: Never output `None` or `Invalid`. Always output a concrete computed value or simplified symbolic expression.
6. Print ONLY the final validated answer in LaTeX boxed format:
   print(f"\\\\boxed{{{final_answer}}}")"""

# Prompt cho Chain-of-Thought (CoT)
COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""

# Prompt cho Zero-shot Direct
DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""

SYSTEM_PROMPTS = {
    "Direct": DIRECT_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
    "SymCode": SYMCODE_SYSTEM_PROMPT,
    "SymReasoner": SYMREASONER_SYSTEM_PROMPT,
    "SymPlanner": SYMPLANNER_SYSTEM_PROMPT,
}


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """
    Xây dựng danh sách thông điệp ChatML cho phương pháp benchmark tương ứng.
    """
    if method == "SymPlanner":
        system_content = SYMPLANNER_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "SymReasoner":
        system_content = SYMREASONER_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "SymCode":
        system_content = SYMCODE_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "CoT":
        system_content = COT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"
    elif method == "Direct":
        system_content = DIRECT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"
    else:
        system_content = DIRECT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def build_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Xây dựng thông điệp tự sửa lỗi (Self-Debugging) cho SymCode với phản hồi chi tiết:
    Traceback thực thi lỗi, đáp án ứng viên trích xuất được và chẩn đoán từ bộ kiểm chứng toán học độc lập.
    """
    feedback_lines = []
    
    if execution_status != "success":
        feedback_lines.append(f"### Execution Status: {execution_status.upper()}")
        if error_tb:
            clean_tb = str(error_tb).strip()
            if len(clean_tb) > 800:
                clean_tb = clean_tb[-800:]
            feedback_lines.append(f"Traceback:\n```\n{clean_tb}\n```")
    else:
        feedback_lines.append("### Execution Status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        cand_short = str(candidate_answer)[:200]
        feedback_lines.append(f"Candidate Answer extracted: `{cand_short}`")
        
    if verification_feedback:
        verif_short = str(verification_feedback)[:800]
        feedback_lines.append(f"Verification Feedback ({verification_status.upper()}):\n{verif_short}")

    feedback_text = "\n\n".join(feedback_lines)

    code_snippet = str(prev_code).strip()
    if len(code_snippet) > 1500:
        code_snippet = code_snippet[:1500]

    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{code_snippet}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            "Please carefully review the diagnosis above. First, briefly explain the root cause and your correction plan. "
            "Then, output the complete corrected Python script enclosed in a single ```python ... ``` block."
        )}
    ]


def build_symreasoner_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Xây dựng thông điệp tự sửa lỗi (Self-Debugging) cho SymReasoner theo framework Constraint-Grounded Synthesis (CGS).
    Bao gồm chẩn đoán lỗi thực thi, kiểm tra ràng buộc nội sinh và hướng dẫn khắc phục SyntaxError / lọc nghiệm ngoại lai.
    """
    feedback_lines = []
    
    if execution_status != "success":
        feedback_lines.append(f"### Execution Status: {execution_status.upper()}")
        if error_tb:
            clean_tb = str(error_tb).strip()
            if len(clean_tb) > 800:
                clean_tb = clean_tb[-800:]
            feedback_lines.append(f"Traceback:\n```\n{clean_tb}\n```")
    else:
        feedback_lines.append("### Execution Status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        cand_short = str(candidate_answer)[:200]
        feedback_lines.append(f"Candidate Answer extracted: `{cand_short}`")
        
    if verification_feedback:
        verif_short = str(verification_feedback)[:800]
        feedback_lines.append(f"Verification Feedback ({verification_status.upper()}):\n{verif_short}")

    feedback_text = "\n\n".join(feedback_lines)

    code_snippet = str(prev_code).strip()
    if len(code_snippet) > 1500:
        code_snippet = code_snippet[:1500]

    symreasoner_hint = (
        "Guidance: If execution encountered a SyntaxError, remove all markdown formatting/prose from inside the code block. "
        "If execution failed with AttributeError/TypeError or outputted 'None'/'Invalid', simplify your script: compute the answer directly, "
        "ensure `check_constraints` uses try-except and does NOT reject valid solutions or overwrite the answer with None, and always print the final answer in \\boxed{}."
    )

    return [
        {"role": "system", "content": SYMREASONER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{code_snippet}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            f"{symreasoner_hint}\n\n"
            "Please carefully review the diagnosis above. First, briefly explain the root cause and your correction plan. "
            "Then, output the complete corrected Python script enclosed in a single ```python ... ``` block."
        )}
    ]


def build_symplanner_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Xây dựng thông điệp tự sửa lỗi (Guarded Repair) cho SymPlanner (Divide-and-Plan Neurosymbolic Synthesis).
    Bao gồm chẩn đoán lỗi thực thi, trạng thái kiểm chứng và hướng dẫn sửa lỗi theo Stage 1 Constraints.
    """
    feedback_lines = []
    
    if execution_status != "success":
        feedback_lines.append(f"### Execution Status: {execution_status.upper()}")
        if error_tb:
            clean_tb = str(error_tb).strip()
            if len(clean_tb) > 800:
                clean_tb = clean_tb[-800:]
            feedback_lines.append(f"Traceback:\n```\n{clean_tb}\n```")
    else:
        feedback_lines.append("### Execution Status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        cand_short = str(candidate_answer)[:200]
        feedback_lines.append(f"Candidate Answer extracted: `{cand_short}`")
        
    if verification_feedback:
        verif_short = str(verification_feedback)[:800]
        feedback_lines.append(f"Verification Feedback ({verification_status.upper()}):\n{verif_short}")

    feedback_text = "\n\n".join(feedback_lines)

    code_snippet = str(prev_code).strip()
    if len(code_snippet) > 1500:
        code_snippet = code_snippet[:1500]

    symplanner_hint = (
        "Guarded Repair: If execution failed with a SyntaxError, ensure your code block contains NO markdown headers or leaked prose. "
        "If verification failed or execution printed 'None', review your Stage 1 Constraints, ensure your Stage 3 code directly computes "
        "the final answer, filters extraneous roots safely, and always prints the result in \\boxed{}."
    )

    return [
        {"role": "system", "content": SYMPLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{code_snippet}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            f"{symplanner_hint}\n\n"
            "Please carefully review the diagnosis above. First, briefly explain the root cause and your correction plan. "
            "Then, output the complete corrected Python script enclosed in a single ```python ... ``` block."
        )}
    ]



