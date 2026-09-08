# Qwen/Qwen3-4B-Instruct-2507 / math500 / smoke

Common completed problems: 8/8. Complete.

Profile: controlled; SymPlan variant: full. Controlled = matched zero-shot + repair; source = source-aligned prompts, PaL 8-shot, no repair. These are adaptations to the evaluated model, not original-paper score reproductions; see METHOD_RESEARCH.md.

| Method | Correct/N | Accuracy % | First attempt accuracy % | ESR % | Output tokens | Input tokens | Calls | Seconds/problem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PaL | 8/8 | 100.00 | 100.00 | 100.00 | 245.4 | 304.8 | 1.00 | 13.62 |
| PoT | 8/8 | 100.00 | 100.00 | 100.00 | 354.5 | 326.8 | 1.00 | 19.51 |
| SymCode | 8/8 | 100.00 | 100.00 | 100.00 | 367.1 | 324.8 | 1.00 | 20.26 |
| SymPlan | 7/8 | 87.50 | 87.50 | 87.50 | 845.8 | 1758.6 | 3.12 | 46.19 |

Paired comparisons: SymPlan minus each baseline; bootstrap intervals are unadjusted.

- PaL: -12.50 pp, 95% CI [-37.50, 0.00], wins/losses 0/1, p=1.00000, Holm p=1.00000
- PoT: -12.50 pp, 95% CI [-37.50, 0.00], wins/losses 0/1, p=1.00000, Holm p=1.00000
- SymCode: -12.50 pp, 95% CI [-37.50, 0.00], wins/losses 0/1, p=1.00000, Holm p=1.00000
