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

# Prompt cho Ablation 1 - SymExtract: Only Stage 1 DIVIDE (State Extraction) -> Stage 3 EXECUTE
SYMEXTRACT_SYSTEM_PROMPT = """You are an expert mathematical reasoner and symbolic computation specialist applying the Divide-and-Plan Neurosymbolic framework (Ablation Study: State Extraction Only).

To solve the problem accurately, follow these two structured steps:

### Stage 1: DIVIDE (Problem State & Constraint Extraction)
Deconstruct the problem into explicit state components:
- Target Unknown: The exact quantity, value, or simplified algebraic expression to compute.
- Given Quantities: Known constants, given parameters, and relationships.
- Domain Constraints: Explicit mathematical and physical boundaries (e.g., positive real numbers, integer constraints, non-zero denominators).

### Stage 2: EXECUTE (Guarded SymCode Generation)
Translate the extracted state components into a complete, self-contained Python script enclosed in a single ```python ... ``` block:
1. Import SymPy as `import sympy as sp`.
2. Define variables with appropriate assumptions based on Stage 1 constraints (e.g., `sp.symbols('x', positive=True, real=True)`).
3. Formulate and solve the equations symbolically using `sp.solve(...)` or direct computation.
4. Filter extraneous roots using Stage 1 constraints.
5. Print ONLY the final validated answer in LaTeX boxed format:
   print(f"\\\\boxed{{{final_answer}}}")"""

# Prompt cho Ablation 2 - SymPlan: Only Stage 2 PLAN (Strategy Formulation) -> Stage 3 EXECUTE
SYMPLAN_SYSTEM_PROMPT = """You are an expert mathematical reasoner and symbolic computation specialist applying the Divide-and-Plan Neurosymbolic framework (Ablation Study: Planner Only).

To solve the problem accurately, follow these two structured steps:

### Stage 1: PLAN (Algorithmic Solution Strategy)
Outline the step-by-step symbolic derivation procedure:
1. Define the system of algebraic/symbolic equations.
2. Specify the exact solving strategy (e.g., substitution, matrix reduction, `sp.solve`, or direct algebraic simplification).
3. Plan how to validate candidate solutions to eliminate extraneous roots.

### Stage 2: EXECUTE (Guarded SymCode Generation)
Translate your plan into a complete, self-contained Python script enclosed in a single ```python ... ``` block:
1. Import SymPy as `import sympy as sp`.
2. Define variables with appropriate assumptions (e.g., `sp.symbols('x', positive=True, real=True)`).
3. Formulate and solve the equations symbolically using `sp.solve(...)` or direct computation.
4. Filter extraneous roots based on your plan.
5. Print ONLY the final validated answer in LaTeX boxed format:
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
    "SymExtract": SYMEXTRACT_SYSTEM_PROMPT,
    "SymPlan": SYMPLAN_SYSTEM_PROMPT,
    "SymPlanner": SYMPLANNER_SYSTEM_PROMPT,
}


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """
    Xây dựng danh sách thông điệp ChatML cho phương pháp benchmark tương ứng.
    """
    if method == "SymPlanner":
        system_content = SYMPLANNER_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "SymExtract":
        system_content = SYMEXTRACT_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "SymPlan":
        system_content = SYMPLAN_SYSTEM_PROMPT
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


def build_symextract_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Xây dựng thông điệp tự sửa lỗi cho SymExtract (Ablation Study: State Extraction Only).
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
        {"role": "system", "content": SYMEXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{code_snippet}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            "Review your Stage 1 extracted state components and domain constraints. "
            "Fix any errors in your Stage 2 Python script, ensure extraneous roots are filtered, and print the result in \\boxed{}."
        )}
    ]


def build_symplan_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Xây dựng thông điệp tự sửa lỗi cho SymPlan (Ablation Study: Planner Only).
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
        {"role": "system", "content": SYMPLAN_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{code_snippet}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            "Review your Stage 1 strategy plan. Fix any errors in your Stage 2 Python script, "
            "ensure equations are correctly formulated and solved, and print the result in \\boxed{}."
        )}
    ]




