# Kaggle LLM Reasoning Benchmark Suite (Qwen2.5-7B-Instruct)

Bộ thử nghiệm đánh giá năng lực suy luận toán học chuẩn hóa (Reasoning Benchmark) cho mô hình ngôn ngữ lớn **Qwen2.5-7B-Instruct** trên các tập dữ liệu **MATH-500** và **GSM8K** với 3 phương pháp cốt lõi:

1. **Direct**: Zero-shot Direct Answering.
2. **CoT (Chain-of-Thought)**: Zero-shot Step-by-Step Natural Language Reasoning (Wei et al., NeurIPS 2022).
3. **SymCode**: Neurosymbolic Equation Solving với SymPy (ACL 2026) tích hợp vòng lặp tự sửa lỗi (Self-Debugging / Verifier Loop) với phản hồi từ Traceback thực thi và Bộ kiểm chứng toán học độc lập (Independent Mathematical Verifier).

---

## 📂 Cấu trúc thư mục `kaggle/`

```text
kaggle/
├── data/                       # Dữ liệu benchmark đã phân loại theo Subject & Level 1-5
│   ├── gsm8k/                  # GSM8K (test.jsonl: phân chia 6 dạng bài & Level 1-5)
│   └── math500/                # MATH-500 (test.jsonl: phân chia 7 subjects & Level 1-5)
├── method/                     # Source code module hóa các phương pháp
│   ├── __init__.py             # Exports các hàm và class chính
│   ├── prompts.py              # System prompts & ChatML message builders
│   ├── extractor.py            # Regex trích xuất \boxed{}, code python, chuẩn hóa LaTeX
│   ├── sandbox.py              # In-memory fast sandbox (bảo vệ timeout, thực thi SymPy an toàn)
│   ├── verifier.py             # Bộ kiểm chứng toán học độc lập (không dùng ground truth)
│   ├── model.py                # LLMRunner (Load Qwen2.5-7B-Instruct 4-bit qua bitsandbytes)
│   └── evaluator.py            # Runner cho các baselines & tổng hợp metrics theo Subject × Difficulty
├── inference.ipynb             # Notebook chính để chạy benchmark trên Kaggle GPU
├── requirements.txt            # Thư viện phụ thuộc (transformers, accelerate, bitsandbytes, sympy, ...)
└── README.md                   # Tài liệu hướng dẫn & mô tả thực nghiệm
```

---

## 🧪 Thiết kế thực nghiệm & Tính công bằng (Fairness & Protocol)

| Tiêu chí | Quy chuẩn thực nghiệm |
| :--- | :--- |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` (Quantization: 4-bit NF4 via `bitsandbytes`, `bfloat16`/`float16`) |
| **Decoding** | `temperature = 0.0` (Greedy Search), `max_new_tokens = 1024` |
| **Dữ liệu & Thứ tự** | Giữ nguyên thứ tự câu hỏi gốc giữa tất cả các phương pháp |
| **Tool Access SymCode** | Python + `sympy` (Symbolic Algebra, Calculus, Number Theory, Geometry). |
| **SymCode Verifier** | Kiểm tra tính nhất quán symbolic, domain constraints, residual không dùng ground truth. Tối đa 2 lần retry (`max_retries = 2`). |
| **Đánh giá (Metrics)** | Exact Match (EM) trên `\boxed{...}`, chuẩn hóa phân số, tọa độ, và symbolic equivalence qua SymPy. |

---

## 📊 Phân rã Metrics (Multi-Dimensional Breakdown)

Kết quả sau khi chạy được tự động lưu ra file JSON (`math500_lvl1_3_results.json` hoặc `gsm8k_results.json`) với đầy đủ các chiều phân tích:

1. **Overall Accuracy (Exact Match %)**
2. **Accuracy by Subject**: `Algebra`, `Number Theory`, `Precalculus`, `Geometry`, `Intermediate Algebra`, `Counting & Probability`, `Prealgebra`, `Arithmetic`, `Percentages & Finance`, `Measurement & Time`...
3. **Accuracy by Difficulty Level**: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`.
4. **Accuracy by Subject × Difficulty**: Ma trận chi tiết theo từng chủ đề và độ khó.
5. **Execution Success Rate**: Tỷ lệ code chạy thành công không bị crash/timeout.
6. **Verification Success Rate**: Tỷ lệ đáp án thỏa mãn bộ kiểm chứng độc lập.
7. **Average Generated Tokens**: Tổng số token trung bình sinh ra cho mỗi bài toán.
8. **Average Attempts**: Số lượt sinh trung bình (cho SymCode).

---

## 🚀 Hướng dẫn chạy trên Kaggle

1. **Upload**: Nén zip thư mục `kaggle` hoặc upload qua Kaggle Datasets.
2. **Chọn Accelerator**: Chọn **GPU T4 x2**, **P100**, hoặc **A100**.
3. **Chạy Notebook `inference.ipynb`**:
   - Tại **Cell 3**, chọn dataset (`math500` hoặc `gsm8k`), chọn danh sách phương pháp (`Direct`, `CoT`, `SymCode`), và số lượng mẫu (`num_samples`).
   - Bấm **Run All**.
   - Notebook có tính năng **Auto-Checkpoint** (lưu ngầm liên tục) và **Auto-Resume** (tiếp tục chạy nếu session bị ngắt quãng).
