import json
import os
import sys
from collections import defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

filepath = "test_compare_n5_lvl5.json"
if not os.path.exists(filepath):
    print("File not found:", filepath)
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

sp_items = data.get("results", {}).get("SymPlanner", [])
total = len(sp_items)
correct_items = [x for x in sp_items if x.get("is_correct")]
wrong_items = [x for x in sp_items if not x.get("is_correct")]

print("=" * 100)
print(f"BÁO CÁO PHÂN TÍCH LỖI DỰA TRÊN {total} MẪU LEVEL 5")
print(f"Số mẫu đúng: {len(correct_items)}/{total} ({len(correct_items)/total*100:.2f}%)")
print(f"Số mẫu sai  : {len(wrong_items)}/{total} ({len(wrong_items)/total*100:.2f}%)")
print("=" * 100 + "\n")

categories = defaultdict(list)

for idx, x in enumerate(sp_items, 1):
    if x.get("is_correct"):
        continue

    gt = str(x.get("ground_truth", "")).strip()
    pred = str(x.get("predicted", "")).strip()
    status = x.get("execution_status")
    attempts = x.get("attempts", 1)
    fb = str(x.get("verification_feedback", "") or "")
    tb = str(x.get("traceback", "") or "")
    subj = x.get("subject", "Unknown")
    prob = x.get("problem", "")

    record = {
        "idx": idx,
        "subject": subj,
        "gt": gt,
        "pred": pred,
        "attempts": attempts,
        "status": status,
        "problem": prob,
        "feedback": fb,
        "traceback": tb
    }

    # Phân loại nguyên nhân sai
    if pred == "None" or status != "success" or "Error" in tb or "Error" in fb:
        if "RecursionError" in tb or "RecursionError" in fb:
            categories["1. Recursion / Call Depth Limit"].append(record)
        elif "Timeout" in status or "Timeout" in fb or "15" in fb:
            categories["2. Execution Timeout (SymPy Solve >15s)"].append(record)
        elif "KeyError" in tb or "IndexError" in tb or "TypeError" in tb:
            categories["3. Python Runtime Exception (Key/Index/Type Error)"].append(record)
        else:
            categories["4. Code Execution Failed (No Boxed Output)"].append(record)
    else:
        # Code chạy ra đáp án nhưng bị sai với Ground Truth
        if "CRootOf" in pred or "atan" in pred or "cos(" in pred or "**(" in pred:
            categories["5. Unsimplified Complex / Radical SymPy Form"].append(record)
        elif "Interval" in pred or "lambda" in pred or "Matrix" in pred or "a6" in pred:
            categories["6. Formatting Mismatch (Interval / Set / Symbolic Tuple)"].append(record)
        else:
            categories["7. Mathematical Reasoning / Model Logic Error"].append(record)

for cat_name, items in categories.items():
    print("=" * 100)
    print(f"{cat_name} (Số lượng: {len(items)})")
    print("=" * 100)
    for r in items:
        print(f"📌 Mẫu #{r['idx']} | Môn: {r['subject']} | Attempts: {r['attempts']} | ExecStatus: {r['status']}")
        prob_snippet = r['problem'] if len(r['problem']) <= 160 else r['problem'][:160] + "..."
        print(f"  - Đề bài     : {prob_snippet}")
        print(f"  - GroundTruth: {r['gt']}")
        print(f"  - Predicted  : {r['pred']}")
        if r['feedback']:
            fb_clean = r['feedback'].replace("\n", " ")
            if len(fb_clean) > 150:
                fb_clean = fb_clean[:150] + "..."
            print(f"  - Feedback   : {fb_clean}")
        print("-" * 100)

print("\n" + "=" * 100)
print("BẢNG TỔNG HỢP PHÂN LOẠI NGUYÊN NHÂN LỖI")
print("=" * 100)
for cat_name, items in categories.items():
    print(f"{cat_name:<60} | {len(items):>2} mẫu ({len(items)/len(wrong_items)*100:.1f}%)")
print("=" * 100)
