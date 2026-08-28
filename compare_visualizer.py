"""
Script trực quan hóa và so sánh đối chiếu kết quả giữa Baseline (CoT / SymCode) và SymPlanner trên MATH-500 Level 5.
Hỗ trợ tùy chỉnh đường dẫn file baseline (--baseline-file), tên baseline (--baseline-name) và số mẫu cần kiểm tra (-n).
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

# Cấu hình encoding stdout cho Windows Console để hỗ trợ Tiếng Việt & Emoji
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trực quan hóa và so sánh đối chiếu kết quả CoT Baseline vs SymPlanner."
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=20,
        help="Số lượng mẫu cần kiểm tra/trực quan hóa (chèn 0 hoặc âm để xem toàn bộ)."
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        default=os.path.join("results", "math500", "cot", "level_5", "results.json"),
        help="Đường dẫn file JSON kết quả Baseline (mặc định: CoT Level 5)."
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="CoT",
        help="Tên của phương pháp Baseline để hiển thị (ví dụ: CoT, SymCode, Direct)."
    )
    parser.add_argument(
        "--symplanner-file",
        type=str,
        default="test_compare_n5_lvl5.json",
        help="Đường dẫn file JSON kết quả SymPlanner nâng cấp."
    )
    parser.add_argument(
        "--only-diff",
        action="store_true",
        default=False,
        help="Chỉ hiển thị các câu có kết quả khác nhau (SymPlanner thắng hoặc Baseline thắng)."
    )
    parser.add_argument(
        "--export-html",
        type=str,
        default=None,
        help="Xuất báo cáo trực quan dạng trang web HTML."
    )
    return parser.parse_args()


def load_results_file(filepath: str, preferred_method: str = "") -> List[Dict[str, Any]]:
    """Tải và trích xuất danh sách kết quả từ file JSON (hỗ trợ cả dạng list [] và dict {})."""
    if not os.path.exists(filepath):
        print(f"[WARN] Không tìm thấy file kết quả tại: {filepath}")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            results_dict = data.get("results", {})
            if preferred_method and preferred_method in results_dict:
                return results_dict[preferred_method]
            for k, v in results_dict.items():
                if isinstance(v, list):
                    return v
        return []
    except Exception as e:
        print(f"[ERROR] Không thể đọc file {filepath}: {e}")
        return []


def main():
    args = parse_args()
    base_name = args.baseline_name

    baseline_items = load_results_file(args.baseline_file, preferred_method=base_name)
    symplanner_items = load_results_file(args.symplanner_file, preferred_method="SymPlanner")

    if not symplanner_items:
        print(f"[ERROR] File SymPlanner rỗng hoặc không đúng định dạng: {args.symplanner_file}")
        sys.exit(1)

    # Đóng gói Baseline thành Dictionary theo câu hỏi để tìm kiếm Nhanh O(1)
    baseline_dict = {item.get("problem", "").strip(): item for item in baseline_items if item.get("problem")}

    # Chọn số lượng mẫu xử lý
    total_available = len(symplanner_items)
    if args.num_samples is not None and args.num_samples > 0:
        limit = min(args.num_samples, total_available)
        eval_items = symplanner_items[:limit]
    else:
        limit = total_available
        eval_items = symplanner_items

    print("=" * 100)
    print(f"{f'BAO CAO SO SANH TRUC QUAN: {base_name.upper()} vs SYMPLANNER (LEVEL 5)':^100}")
    print("=" * 100)
    print(f"[INFO] So mau kiem tra: {limit}/{total_available}")
    print(f"[INFO] File Baseline ({base_name}): {args.baseline_file}")
    print(f"[INFO] File SymPlanner: {args.symplanner_file}")
    print("=" * 100 + "\n")

    base_correct_count = 0
    sp_correct_count = 0
    base_matched_count = 0
    
    comparisons = []

    for idx, sp_item in enumerate(eval_items, 1):
        prob = sp_item.get("problem", "").strip()
        gt = sp_item.get("ground_truth", "")
        subj = sp_item.get("subject", "Unknown")
        
        sp_pred = sp_item.get("predicted")
        sp_corr = bool(sp_item.get("is_correct", False))
        sp_tokens = sp_item.get("generated_tokens", 0)
        sp_attempts = sp_item.get("attempts", 1)
        planner_note = sp_item.get("planner_note", "")

        base_item = baseline_dict.get(prob)
        if base_item:
            base_matched_count += 1
            base_pred = base_item.get("predicted")
            base_corr = bool(base_item.get("is_correct", False))
            base_tokens = base_item.get("generated_tokens", 0)
        else:
            base_pred = "N/A (Chua tim thay)"
            base_corr = False
            base_tokens = 0

        if base_corr:
            base_correct_count += 1
        if sp_corr:
            sp_correct_count += 1

        # Xác định trạng thái so sánh
        if sp_corr and not base_corr:
            status = "SYMPLANNER WINS"
            badge = f"[SymPlanner THANG]"
        elif base_corr and not sp_corr:
            status = f"{base_name.upper()} WINS"
            badge = f"[{base_name} THANG]"
        elif sp_corr and base_corr:
            status = "BOTH CORRECT"
            badge = "[CA 2 DUNG]"
        else:
            status = "BOTH WRONG"
            badge = "[CA 2 SAI]"

        record = {
            "idx": idx,
            "problem": prob,
            "subject": subj,
            "ground_truth": gt,
            "base_pred": base_pred,
            "base_corr": base_corr,
            "base_tokens": base_tokens,
            "sp_pred": sp_pred,
            "sp_corr": sp_corr,
            "sp_tokens": sp_tokens,
            "sp_attempts": sp_attempts,
            "planner_note": planner_note,
            "status": status,
            "badge": badge
        }
        comparisons.append(record)

    # Hiển thị từng mẫu chi tiết
    displayed_count = 0
    for rec in comparisons:
        if args.only_diff and rec["sp_corr"] == rec["base_corr"]:
            continue
            
        displayed_count += 1
        print(f"----------------------------------------------------------------------------------------------------")
        print(f"[*] MAU #{rec['idx']} | Mon hoc: {rec['subject']} | Trang thai: {rec['badge']}")
        print(f"----------------------------------------------------------------------------------------------------")
        prob_short = rec['problem'] if len(rec['problem']) <= 250 else rec['problem'][:250] + "..."
        print(f"De bai: {prob_short}")
        print(f"Ground Truth: {rec['ground_truth']}")
        print(f"----------------------------------------------------------------------------------------------------")
        
        base_icon = "[DUNG]" if rec['base_corr'] else "[SAI]"
        sp_icon = "[DUNG]" if rec['sp_corr'] else "[SAI]"
        
        print(f"{base_name:<11} {base_icon:<6} : Predicted = '{rec['base_pred']}' (Tokens: {rec['base_tokens']})")
        print(f"SymPlanner  {sp_icon:<6} : Predicted = '{rec['sp_pred']}' (Attempts: {rec['sp_attempts']}, Tokens: {rec['sp_tokens']})")
        
        if rec['planner_note'] and len(str(rec['planner_note'])) > 10:
            note_str = str(rec['planner_note']).strip().replace("\n", " ")
            if len(note_str) > 180:
                note_str = note_str[:180] + "..."
            print(f"Planner JSON Note: {note_str}")
        print("")

    # Bảng tổng kết chỉ số
    print("=" * 100)
    print(f"{'BANG TONG NANG LUC DUA TREN ' + str(limit) + ' MAU KICH THUOC':^100}")
    print("=" * 100)
    base_acc = (base_correct_count / limit) * 100.0 if limit > 0 else 0
    sp_acc = (sp_correct_count / limit) * 100.0 if limit > 0 else 0
    
    print(f"{'Phuong phap':<22} | {'Dung / Tong':<15} | {'Accuracy (%)':<15} | {'Mau doi chieu khop de'}")
    print("-" * 100)
    print(f"{f'{base_name} (Baseline)':<22} | {f'{base_correct_count}/{limit}':<15} | {base_acc:<14.2f}% | {base_matched_count}/{limit}")
    print(f"{'SymPlanner (Nang cap)':<22} | {f'{sp_correct_count}/{limit}':<15} | {sp_acc:<14.2f}% | {limit}/{limit}")
    print("=" * 100 + "\n")

    # Xuất báo cáo HTML nếu có yêu cầu
    if args.export_html:
        export_html_report(comparisons, args.export_html, base_acc, sp_acc, limit, base_name)


def export_html_report(comparisons: List[Dict[str, Any]], filepath: str, base_acc: float, sp_acc: float, total: int, base_name: str):
    """Xuất file báo cáo HTML giao diện đẹp để mở bằng trình duyệt web."""
    cards_html = []
    for c in comparisons:
        base_cls = "pass" if c["base_corr"] else "fail"
        sp_cls = "pass" if c["sp_corr"] else "fail"
        badge_cls = "wins" if "WINS" in c["status"] else ("both-pass" if c["sp_corr"] and c["base_corr"] else "both-fail")
        
        card = f"""
        <div class="card {badge_cls}">
            <div class="card-header">
                <span class="idx">Mẫu #{c['idx']}</span>
                <span class="subject">{c['subject']}</span>
                <span class="badge">{c['badge']}</span>
            </div>
            <div class="problem"><strong>Đề bài:</strong> {c['problem']}</div>
            <div class="gt"><strong>Ground Truth:</strong> <code>{c['ground_truth']}</code></div>
            <div class="results-grid">
                <div class="res-box {base_cls}">
                    <h4>{base_name} Baseline</h4>
                    <p><strong>Dự đoán:</strong> <code>{c['base_pred']}</code></p>
                    <p><strong>Trạng thái:</strong> { '✅ ĐÚNG' if c['base_corr'] else '❌ SAI' }</p>
                    <p><small>Tokens: {c['base_tokens']}</small></p>
                </div>
                <div class="res-box {sp_cls}">
                    <h4>SymPlanner (Nâng cấp)</h4>
                    <p><strong>Dự đoán:</strong> <code>{c['sp_pred']}</code></p>
                    <p><strong>Trạng thái:</strong> { '✅ ĐÚNG' if c['sp_corr'] else '❌ SAI' }</p>
                    <p><small>Attempts: {c['sp_attempts']} | Tokens: {c['sp_tokens']}</small></p>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Báo cáo So sánh {base_name} vs SymPlanner Level 5</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; color: #333; }}
        h1 {{ text-align: center; color: #2c3e50; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 20px; }}
        .summary-container {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; border-radius: 8px; padding: 20px; min-width: 200px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .stat-card h3 {{ margin: 0; color: #7f8c8d; font-size: 14px; text-transform: uppercase; }}
        .stat-card .val {{ font-size: 32px; font-weight: bold; margin: 10px 0; color: #2c3e50; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 6px solid #ccc; }}
        .card.wins {{ border-left-color: #27ae60; }}
        .card.both-pass {{ border-left-color: #2980b9; }}
        .card.both-fail {{ border-left-color: #e74c3c; }}
        .card-header {{ display: flex; align-items: center; gap: 15px; margin-bottom: 10px; }}
        .idx {{ font-weight: bold; background: #34495e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .subject {{ background: #ecf0f1; color: #2c3e50; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
        .badge {{ font-weight: bold; font-size: 13px; color: #2c3e50; }}
        .problem {{ font-size: 15px; margin-bottom: 10px; background: #fafafa; padding: 10px; border-radius: 4px; line-height: 1.5; }}
        .gt {{ font-size: 14px; margin-bottom: 15px; color: #8e44ad; }}
        .results-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        .res-box {{ padding: 12px; border-radius: 6px; border: 1px solid #ddd; }}
        .res-box.pass {{ background: #e8f8f5; border-color: #a3e4d7; }}
        .res-box.fail {{ background: #fdf2e9; border-color: #fad7a0; }}
        .res-box h4 {{ margin: 0 0 8px 0; font-size: 14px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: Consolas, monospace; color: #c7254e; }}
    </style>
</head>
<body>
    <h1>Báo cáo So sánh Trực quan {base_name} vs SymPlanner</h1>
    <div class="subtitle">MATH-500 Level 5 (Quy mô: {total} mẫu)</div>
    
    <div class="summary-container">
        <div class="stat-card">
            <h3>{base_name} Baseline</h3>
            <div class="val" style="color: #e67e22;">{base_acc:.2f}%</div>
        </div>
        <div class="stat-card">
            <h3>SymPlanner (Nâng cấp)</h3>
            <div class="val" style="color: #27ae60;">{sp_acc:.2f}%</div>
        </div>
    </div>

    <div class="cards-list">
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[SUCCESS] Đã xuất báo cáo trang web HTML thành công tại: {filepath}")
    except Exception as e:
        print(f"[ERROR] Không thể xuất báo cáo HTML: {e}")


if __name__ == "__main__":
    main()
