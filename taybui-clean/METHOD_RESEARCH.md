# Four program-aided methods: provenance and controlled adaptations

Protocol v3, reviewed 2026-09-08. The active entry point is `python -m paperbench.run`.
The exact method IDs are `PaL`, `PoT`, `SymCode`, `SymPlan`.
These are controlled zero-shot adaptations, not reproductions of published scores.
Controlled prompts are independently written. The source profile includes the
original eight PaL demonstrations under their Apache-2.0 license, pinned locally.

## PaL

Sources: [PAL paper](https://arxiv.org/abs/2211.10435),
[official math prompt](https://github.com/reasoning-machines/pal/blob/main/pal/prompt/math_prompts.py).

The official math examples generate `solution()` with meaningful quantity names,
executable intermediate operations and a returned result. We retain that structure.
Our adaptation uses zero-shot chat instructions instead of the original few-shot
completion prompt. The generated program explicitly calls the function and prints
boxed LaTeX. SymPy is available for symbolic MATH answers, as it is for every method.
The shared failure-repair wrapper is an experimental extension, not original PAL.

## PoT

Sources: [PoT paper](https://arxiv.org/abs/2211.12588),
[official zero-shot GSM8K runner](https://github.com/TIGER-AI-Lab/Program-of-Thoughts/blob/main/run_gsm8k_zs.py),
[official repository](https://github.com/TIGER-AI-Lab/Program-of-Thoughts).

PoT expresses intermediate computation as programs; its zero-shot GSM8K prompt
requests a stepwise `solver()` function starting with variable definitions.
Our prompt follows this branch and permits symbolic equations where needed.
We use chat generation, complete programs and boxed output rather than completing
a function prefix and converting the result to a float. Greedy single-sample
inference replaces self-consistency. Shared repair is an extension.
PaL and PoT are closely related; function names alone do not establish a new
algorithm or imply a large empirical difference under this adaptation.

## SymCode

Source: [published paper, sections 3.1–3.2](https://aclanthology.org/2026.findings-eacl.76/).

The implementation preserves SymPy formulation, domain assumptions, mathematical
step comments, substitution and constraint checks. It no longer uses the earlier
repo's generic direct-SymPy prompt as a stand-in. The paper distinguishes the base
SymCode prompt from SymCode+ with execution-triggered debugging. Our `SymCode`
label denotes the base prompt under the shared repair protocol; retry results
must be described as a controlled SymCode+-style extension, not original SymCode.
We standardize formatting, resources and budgets rather than reproduce its models
or retry settings. We do not force two independent algorithms or prohibit exact
symbolic solving, neither of which is necessary to implement section 3.1.

## SymPlan

Local method: target/domain/relation extraction → concrete algorithm derivation
→ Python/SymPy → execution → failure repair. Each LLM stage receives the original
question. Planning can correct extraction, reduce equations, establish finite
search bounds, filter candidates and specify an original-relation check. Code
must compute the target; extraction/planning text is never an answer fallback.
These are design improvements to evaluate on dev, not evidence of superiority.

## Fairness and reporting

All four methods use the same model/revision, precision, greedy non-thinking mode,
questions, library access, parser, interpreter, timeout, grading and stopping rule.
All have a 3,072 generated-token ceiling, a 1,536-token code/repair call ceiling,
and at most one repair after an execution, assertion, timeout or output failure.
SymPlan pays 256 representation and 384 planning tokens from its own total budget.
These are upper bounds: early success leaves unused tokens and equal ceilings
are not equal token consumption, input tokens, FLOPs or wall time. All calls and
limits are recorded; reports show actual costs and accuracy before/after repair.
The first-attempt result still uses the controlled zero-shot prompts and is not
an original-paper replication. Valid executed output stops every method even if
it is wrong; gold is used only after solve, never to trigger repair or select code.

The fixed suite runs dev, freezes, then tests regardless of which method wins.
There is no winner gate or adaptive budget selection. Three separate SymPlan
component ablations remove extraction, planning, or both under the controlled
protocol. Use a new output directory; old v2 manifests/locks are incompatible.

## Source-aligned sensitivity profile (additional table)

`--prompt-profile source` enforces zero retries for every method. PaL receives
`MATH_PROMPT` from the pinned upstream file (8 demonstrations); PoT receives a
short instruction reflecting the official zero-shot `solver()` branch. Their
returned functions are invoked by a deterministic output adapter; programs need
not learn the boxed print interface. SymCode retains the independently worded
section-3.1 requirements. SymPlan retains its three-stage design. All methods
receive up to 3,072 output tokens, with SymPlan paying for its intermediate stages.

This profile preserves PaL's example content and function-return execution but
still uses Qwen's chat template, a different backbone, tokenizer, budgets and
symbolic output serialization. PoT's function-completion prefix is expressed as
a complete-program chat instruction. It is source-aligned, not a verbatim
reproduction; it is also not shot-matched. The controlled table remains the
matched zero-shot comparison. No source profile is mislabeled as original scores.

The unmodified PaL file, license and SHA256 provenance are in `paperbench/vendor/pal/`.
SymCode's published boxed-print requirement is implemented with valid escaped
LaTeX braces; no malformed PDF-transcribed format string is copied.
