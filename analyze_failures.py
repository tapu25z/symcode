import json
import os

filepath = "test_compare_n5_lvl5.json"
if not os.path.exists(filepath):
    print("File not found:", filepath)
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

sp_items = data.get("results", {}).get("SymPlanner", [])
total = len(sp_items)
correct = [x for x in sp_items if x.get("is_correct")]
wrong = [x for x in sp_items if not x.get("is_correct")]

print(f"Total SymPlanner samples: {total}")
print(f"Correct: {len(correct)} ({len(correct)/total*100:.2f}%)")
print(f"Wrong: {len(wrong)} ({len(wrong)/total*100:.2f}%)\n")

print("=" * 80)
print("ANALYSIS OF FAILED SAMPLES")
print("=" * 80)

failure_categories = {
    "unevaluated_symbolic_vars": [],
    "syntax_execution_error": [],
    "domain_constraint_violation": [],
    "formatting_mismatch": [],
    "math_logic_error": []
}

for idx, x in enumerate(sp_items, 1):
    if x.get("is_correct"):
        continue
    
    gt = str(x.get("ground_truth", ""))
    pred = str(x.get("predicted", ""))
    status = x.get("execution_status")
    attempts = x.get("attempts")
    fb = str(x.get("verification_feedback", ""))
    subj = x.get("subject", "")
    prob = x.get("problem", "")
    history = x.get("attempt_history", [])

    print(f"\n[SAMPLE #{idx}] Subject: {subj} | Attempts: {attempts} | ExecStatus: {status}")
    print(f"Ground Truth : {gt}")
    print(f"Predicted    : {pred}")
    print(f"Problem      : {prob[:180]}...")
    print(f"Feedback     : {fb[:200]}...")
    
    # Categorize
    if "unevaluated Python variable name" in fb or "contains none of these target symbols" in fb:
        failure_categories["unevaluated_symbolic_vars"].append((idx, subj, gt, pred))
    elif status != "success" or "SyntaxError" in fb or "TypeError" in fb:
        failure_categories["syntax_execution_error"].append((idx, subj, gt, pred))
    elif "domain" in fb.lower() or "constraint" in fb.lower():
        failure_categories["domain_constraint_violation"].append((idx, subj, gt, pred))
    else:
        failure_categories["math_logic_error"].append((idx, subj, gt, pred))

print("\n" + "=" * 80)
print("FAILURE CATEGORY SUMMARY")
print("=" * 80)
for cat, items in failure_categories.items():
    print(f"Category [{cat}]: {len(items)} cases")
    for idx, subj, gt, pred in items:
        print(f"  - Sample #{idx} [{subj}]: GT='{gt}', Pred='{pred}'")
