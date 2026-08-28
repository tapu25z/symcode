# LLM Reasoning Benchmark Suite: Qwen2.5-Coder-7B-Instruct
## Baselines vs. SymPlan (Ours - Verifiable IR Pipeline)

Hệ thống benchmark đánh giá năng lực suy luận toán học chuẩn hóa cho mô hình ngôn ngữ lớn **Qwen2.5-Coder-7B-Instruct** trên các tập dữ liệu benchmark tiêu chuẩn **MATH-500** và **GSM8K**.

Toàn bộ mã nguồn đã được mô-đun hóa thành các thư mục độc lập dưới `methods/` và tài nguyên dùng chung trong `shared/`:

---

## 1. Cấu trúc thư mục dự án `kaggle2`

```text
kaggle2/
├── README.md                   # Tài liệu hướng dẫn tổng quan và thực thi chi tiết
├── inference.ipynb             # Notebook Kaggle siêu gọn (3 bước: Setup -> Config -> Run)
├── run_notebook.py             # Engine điều phối thực thi và hiển thị kết quả cho Notebook
├── run_benchmark.py            # Script thực thi benchmark qua giao diện dòng lệnh (CLI)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── data/                       # Tập dữ liệu benchmark chuẩn hóa
│   ├── gsm8k/                  # GSM8K (1.320 mẫu test phân loại theo chủ đề và độ khó Level 1-5)
│   │   └── test.jsonl
│   └── math500/                # MATH-500 (500 mẫu test gồm 7 chủ đề và độ khó Level 1-5)
│       └── test.jsonl
├── shared/                     # Bộ công cụ dùng chung cho toàn bộ các phương pháp
│   ├── __init__.py
│   ├── model.py                # LLMRunner (4-bit NF4 bitsandbytes, SDPA attention)
│   ├── sandbox.py              # In-Memory Fast Sandbox thực thi an toàn với timeout
│   ├── extractor.py            # Boxed extraction, AST Sanitizer, exact match
│   ├── data_loader.py          # Nạp dataset, lọc Level 1-5 và lọc Subject
│   └── metrics.py              # Tính toán chỉ số đa chiều và lưu kết quả JSON
└── methods/                    # 4 PHƯƠNG PHÁP ĐƯỢC TÁCH RIÊNG BIỆT:
    ├── direct/                 # Phương pháp 1: Direct Zero-shot (Baseline)
    │   ├── __init__.py
    │   ├── prompts.py          # Direct prompt
    │   └── evaluator.py        # Evaluator cho Direct
    ├── cot/                    # Phương pháp 2: Chain-of-Thought (Baseline)
    │   ├── __init__.py
    │   ├── prompts.py          # CoT prompt
    │   └── evaluator.py        # Evaluator cho CoT
    ├── symcode/                # Phương pháp 3: SymCode tiêu chuẩn (Baseline)
    │   ├── __init__.py
    │   ├── prompts.py          # SymCode code prompt & repair prompt
    │   ├── verifier.py         # Bộ kiểm chứng toán học độc lập
    │   └── evaluator.py        # Evaluator cho SymCode với vòng lặp Self-Debugging
    └── symplan/                # Phương pháp 4: SymPlan (Ours - Verifiable IR Pipeline)
        ├── __init__.py
        ├── problem_ir.py       # ProblemIR contract, schema validator & IR repair
        ├── normalizer.py       # Chuẩn hóa biểu thức, đơn vị SI và Codegen Payload
        ├── prompts.py          # Prompts: Extraction, IR Repair, Codegen, Targeted Repair
        ├── relation_verifier.py# Bidirectional Verifier kiểm tra quan hệ thuận & nghịch
        ├── adapters.py         # Model/Sandbox Adapters với Stage Token Budgets
        ├── scoring.py          # So khớp tương đương đại số (Matrix, Set, Interval, Expression)
        ├── config.py           # Cấu hình biến thể Ablation
        ├── pipeline.py         # SymPlannerIRPipeline orchestration
        └── evaluator.py        # Evaluator cho SymPlan & tính toán IR Diagnostics
```

---

## 2. Hướng dẫn chạy trên Kaggle

Kaggle cung cấp GPU miễn phí (NVIDIA T4 x2 hoặc P100). Gói `kaggle2.zip` ở thư mục gốc đã được đóng gói sẵn để bạn upload nhanh.

### Bước 1: Chuẩn bị mã nguồn
- Sử dụng file `kaggle2.zip` có sẵn ở thư mục gốc.

### Bước 2: Tạo Dataset trên Kaggle
1. Đăng nhập [Kaggle](https://www.kaggle.com/) -> Vào mục **Datasets** -> Chọn **New Dataset**.
2. Đặt tên dataset: `kaggle2-benchmark`.
3. Upload file `kaggle2.zip` -> Nhấn **Create**.

### Bước 3: Tạo và Cấu hình Notebook trên Kaggle
1. Vào mục **Code** -> Chọn **New Notebook**.
2. Bảng điều khiển bên phải (**Notebook options**):
   - **Accelerator**: Chọn **GPU T4 x2** hoặc **GPU P100**.
   - **Internet**: Chọn **Internet On** (bắt buộc để tải weights mô hình từ Hugging Face).
   - **Persistence**: Chọn **Variables and files** (để giữ checkpoint).
3. Nhấn **Add Input** -> Chọn dataset `kaggle2-benchmark`.

### Bước 4: Thực thi Notebook
File `inference.ipynb` chỉ gồm 3 cell code đơn giản:

```python
# Cell 1: Khởi tạo môi trường
from run_notebook import setup_environment
setup_environment()

# Cell 2: Tùy chỉnh cấu hình
DATASET = "math500"                                 # "math500" hoặc "gsm8k"
METHODS = ["Direct", "CoT", "SymCode", "SymPlan"]   # Baselines: Direct, CoT, SymCode | Ours: SymPlan
NUM_SAMPLES = 5                                     # Số lượng mẫu (None = toàn bộ)
FILTER_LEVELS = [1, 2, 3]                           # Lọc Level 1-5 (None = tất cả)

from run_notebook import build_config
config = build_config(
    dataset_name=DATASET,
    methods=METHODS,
    num_samples=NUM_SAMPLES,
    filter_levels=FILTER_LEVELS
)

# Cell 3: Thực thi và Hiển thị bảng kết quả
from run_notebook import run_benchmark_pipeline
results = run_benchmark_pipeline(config)
```

---

## 3. Hướng dẫn chạy qua CLI trên Máy cục bộ / Server

```bash
# 1. Truy cập thư mục kaggle2
cd symcode/kaggle2

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy benchmark mẫu với Baselines và SymPlan (Ours):
python run_benchmark.py --dataset math500 --num-samples 5 --filter-levels 1 2 3 --methods Direct CoT SymCode SymPlan
```
