# Audit before server execution — 2026-09-08

## Method correspondence

| Method | Checked against | Implemented | Remaining adaptation |
|---|---|---|---|
| PaL | Official `pal/prompt/math_prompts.py` | Controlled zero-shot; source profile vendors the original 8-shot prompt and executes `solution()` | Qwen chat backbone, budgets and LaTeX serialization |
| PoT | Official `run_gsm8k_zs.py`, PoT paper | Stepwise `solver()` with meaningful variables; source executor calls returned function | Complete-program chat instruction replaces a completion prefix; no self-consistency |
| SymCode | EACL 2026 paper sections 3.1/3.2 | SymPy, domain assumptions, reasoning comments and executable checks; source disables retry | Independently worded prompt; controlled shared retry is SymCode+-style |
| SymPlan | Local algorithm | Representation, planning, code and failure repair, plus three component ablations | No claim of external-paper reproduction |

Source URLs and PaL revision/checksums are in METHOD_RESEARCH.md and
paperbench/vendor/pal/provenance.json. Paper scores must be attributed to these
explicitly described adaptations, not to a verbatim reproduction.

## Corrections made in this audit

- Replaced winner-gated development workflow with a fixed suite that runs test
  regardless of relative accuracy; no test-based changes to later protocols.
- Froze model revision and runtime identity from dev through test; locked the
  entire supervisor against concurrent launches and incompatible resumes.
- Added source-prompt sensitivity runs and separate component ablations.
- Fixed function-output serialization for Python Fraction using SymPy conversion.
- Added complete-test-only CSV/LaTeX export and checks for protocol/grading errors.
- Added Linux bootstrap, pinned PyTorch CUDA wheel, runtime pip freeze and a
  launch script for smoke → dev → freeze → full test → export.

## Validation and limits

Local CPU suite: 63 tests and 3 subtests passed (including legacy tests); the
paperbench tests also exercise real Python/SymPy execution, timeouts, source
function adapters, repair, total budgets, ablations and table export. GPU model
loading, VRAM, latency and Linux seccomp execution have not been exercised on
this macOS host. Server bootstrap runs paperbench tests again in its own Python
environment and smoke exercises the real model/executor before full runs.

Exact normalized-question comparison found zero PaL demonstration matches in the
prepared MATH/GSM8K dev/test files. This is not a claim of semantic deduplication
or absence of model pretraining contamination. MATH-500 has prior development
exposure in this project and must be disclosed.

No test accuracy is available yet. The implementation cannot guarantee that
SymPlan wins; export preserves losing results and unadjusted confidence intervals.
Holm adjustment covers the three baseline comparisons within each dataset/profile;
it is not a global adjustment across all tables or exploratory claims.
