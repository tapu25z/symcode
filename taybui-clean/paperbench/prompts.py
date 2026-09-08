"""Controlled zero-shot adaptations; provenance and deviations: METHOD_RESEARCH.md."""

METHODS = ("PaL", "PoT", "SymCode", "SymPlan")
BASELINES = METHODS[:-1]

# Common interface and resources, not a common reasoning algorithm.
PROGRAM_CONTRACT = r"""Return one complete executable Python program in a single
```python ... ``` block, without prose outside it. Import every library you use.
Available libraries: sympy, math, fractions, itertools, functools, collections,
decimal, statistics, numpy and scipy. Do not access files, network or processes.
Preserve exact quantities, symbolic answers, tuple order and interval endpoints.
Use rational arithmetic for exact fractions; approximate only when appropriate
for the requested answer. Compute the answer; do not print an unsupported guess.
Your program must call its solver, if it defines one, and print exactly one final
answer in LaTeX boxed form. For example, after importing sympy as sp:
print(r"\boxed{%s}" % sp.latex(sp.sympify(answer)))
For text or base notation, print the appropriate LaTeX string directly.
Do not print intermediate values or placeholders."""

PAL = """Translate the problem into a Python solution() function that returns the
requested answer. Represent the quantities with descriptive variable names.
Express the reasoning as a sequence of executable operations, with short comments
connecting those operations to the problem. Let the Python interpreter perform
the calculations. Call solution(), assign its return value to answer, and print
it using the output contract below.
""" + PROGRAM_CONTRACT

POT = """Write a Python solver() function step by step to answer the question.
First define meaningful variables for the quantities in the problem. Express
intermediate thoughts as assignments and mathematical expressions, so computation
is carried out by Python rather than by mental arithmetic. When an unknown must
be inferred from relations, define symbolic variables and equations and use a
symbolic solver as appropriate. Return the requested answer from solver(), call
it, and print its result using the output contract below.
""" + PROGRAM_CONTRACT

SYMCODE = """Construct a self-contained SymPy program whose code is the mathematical
reasoning trace. Import sympy as sp. Document the variables, constraints and goal
in comments and declare symbols with appropriate domain assumptions. Explain
each mathematical step in code comments and implement the algebraic operations
with meaningful variable names. Substitute candidate solutions into the original
relations, check their domains, and filter invalid roots. Include executable
mathematical assertions where applicable so verification failures raise errors.
""" + PROGRAM_CONTRACT

EXTRACT = """Represent the original problem compactly for a mathematical programmer.
Return five short fields:
TARGET: the exact requested object and output form, not a downstream quantity.
VARIABLES: unknowns with domains; distinguish given symbolic parameters.
RELATIONS: equations, definitions, counts or geometry facts from the problem.
CONSTRAINTS: signs, bounds, integrality, exclusions and ordering.
CHECK: an original relation or invariant that can validate the target/candidates.
Preserve all necessary data and exact fractions. Do not invent assumptions,
solve the problem or write code. Prioritize the target and governing relations
instead of repeating the question. Mark genuinely missing information explicitly."""

PLAN = """Derive a compact executable mathematical algorithm from the original
problem and extracted state. Treat the original problem as authoritative and
correct any extraction error. Write 3-6 concise numbered steps, without Python:
1. Formulate the target and derive the essential equations or reduction.
2. Choose a concrete algorithm: direct formula, exact algebra, recurrence, or
bounded enumeration as appropriate. State sufficient bounds for every search;
do not use guessed cutoffs. Reduce the system before using a general solver.
3. Specify candidate filtering against the original domains and relations.
4. State the final target transformation and a cheap independent mathematical
check from the original problem, including the required answer format.
Intermediate derivations are allowed. Avoid generic steps such as 'solve it',
multiple competing plans, and calculating a final number just to copy into code."""

SYMPLAN = """Implement the supplied mathematical state and algorithm as a concise
Python/SymPy program. The original problem overrides the notes; fix conflicts.
Import sympy as sp. Preserve domain assumptions and exact arithmetic. Implement
the derived reduction before resorting to a large symbolic solve. Use bounded
search only when the bounds are justified; never approximate an infinite sum
with an arbitrary cutoff. Filter extraneous roots using the original relations.
Add a cheap substitution or invariant check where feasible; assertions must test
mathematics, not simply whether a solution exists. If solving fails, do not catch
an exception just to print a guess. Compute and return precisely the requested
target. Keep comments focused on the mathematical steps that need explanation.
""" + PROGRAM_CONTRACT

CODE_PROMPTS = {"PaL": PAL, "PoT": POT, "SymCode": SYMCODE, "SymPlan": SYMPLAN}

REPAIR = """
The previous program failed execution or output extraction. Use the original
problem, previous program and interpreter diagnosis to repair it. Preserve this
method's program structure. Correct mathematical formulation errors as well as
Python errors; do not delete constraints or assertions just to suppress a failure.
Return a complete replacement program. No reference answer is available.
"""


def messages(system, question, context=""):
    return [{"role": "system", "content": system},
            {"role": "user", "content": "ORIGINAL PROBLEM:\n" + question +
             ("\n\n" + context if context else "")}]

# Source-aligned sensitivity run, kept separate from the matched zero-shot table.
# PaL uses upstream demonstrations unchanged; Qwen still uses its chat template.
SOURCE_POT = """Implement a solver() function in Python for this question. Begin
by defining the needed variables, then express the solution step by step in code
and return the answer. Return only the complete program, without calling solver()
or printing: the evaluator will call it. Import libraries that you use."""


def generation_messages(method, profile, stage, system, question, context=""):
    if profile == "source" and stage == "code":
        if method == "PaL":
            from .vendor.pal.math_prompts import MATH_PROMPT
            return [{"role": "system", "content":
                     "Complete the final problem with a Python solution() function. "
                     "Return only its complete code, without calling or printing it. "
                     "Import libraries that you use."},
                    {"role": "user", "content": MATH_PROMPT.format(question=question)}]
        if method == "PoT":
            return messages(SOURCE_POT, question)
    return messages(system, question, context)


def executable_program(code, method, profile):
    """Source function-return adapter; no solving, AST repair or answer inference."""
    if profile == "source" and method in {"PaL", "PoT"}:
        function = "solution" if method == "PaL" else "solver"
        return code + '\nimport sympy as _sp\n_result = ' + function + '()\n' + (
            'print(r"\\boxed{%s}" % (_result if isinstance(_result, str) else _sp.latex(_sp.sympify(_result))))\n')
    return code
