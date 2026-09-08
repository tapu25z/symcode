# Chạy số liệu paper: PaL, PoT, SymCode, SymPlan

Bộ chạy hiện hành là `paperbench/`, protocol **paper_v3**. Runner cũ chỉ để truy vết.
Đọc [METHOD_RESEARCH.md](METHOD_RESEARCH.md) trước khi đặt tên baseline trong paper.

## Model và thiết kế đã chốt

**deepseek-ai/deepseek-coder-1.3b-instruct**, NF4 4-bit, batch 1, một GPU RTX 3090
24GB, greedy decoding, seed 42. Model nhỏ, thiên về sinh code, phù hợp để đo
program-aided inference mà không dùng Qwen3. Model revision được ghim lúc khởi
động suite và giữ nguyên cho mọi run. NF4 dùng đồng nhất cho mọi method và suite.

Cả bốn method lấy đáp án từ Python. Cùng dữ liệu, thư viện, timeout 15 giây/lần,
grading Math-Verify, input tối đa 8.192 token/call, tổng output tối đa 3.072 token/bài.
Mọi token extraction/planning/repair đều tính vào tổng; không cắt đề khi quá context.

| Suite | Prompt / retry | Code cap | Vai trò |
|---|---|---:|---|
| controlled | Zero-shot cho cả 4; cùng tối đa 1 repair khi lỗi | 1.536/call | Bảng so sánh kiểm soát chính |
| source | PaL 8 ví dụ upstream; PoT nhánh zero-shot; không retry cả 4 | 3.072/call, bị chặn bởi ngân sách còn lại | Kiểm tra độ nhạy theo prompt nguồn |
| no-extract | Chỉ SymPlan, bỏ extraction | 1.536/call | Ablation |
| no-plan | Chỉ SymPlan, bỏ planning | 1.536/call | Ablation |
| code-only | Chỉ SymPlan, bỏ cả hai | 1.536/call | Ablation |

SymPlan full dùng extraction tối đa 256 token và planning 384 token. Ablation dùng
nguyên code prompt và chính sách repair, không đặt thành baseline mới trong bảng chính.
Source không shot-matched vì PaL giữ few-shot gốc. Hai bảng trả lời hai câu hỏi khác
nhau; không trộn chúng hoặc chọn bảng nào SymPlan thắng để làm kết quả duy nhất.
Equal output ceilings không đồng nghĩa equal actual tokens, FLOPs hoặc wall time.

## Server và cài đặt

Linux x86_64, Python 3.10–3.12 (khuyến nghị 3.11), NVIDIA driver tương thích CUDA 12.4,
GPU có BF16. Process executor cần root, thư viện `libseccomp.so.2`, checkout root-owned,
mode 700 và venv bên ngoài checkout. Trên Debian/Ubuntu, cài `python3-venv` và
`libseccomp2` bằng package manager nếu thiếu. Chạy từ thư mục dự án đã upload:

```bash
bash scripts/setup_paper_server.sh
```

Script tạo `/opt/symplan-paper-v3-venv`, cài PyTorch 2.6.0 CUDA 12.4 + requirements, chạy
CPU tests, kiểm tra CUDA/BF16/libseccomp, chuẩn bị dữ liệu nếu chưa có manifest,
đặt checkout mode 700 và lưu `paper_runs/server-requirements.lock.txt`.
Dữ liệu và output phải ở bên trong checkout, venv ở ngoài để UID 65534 đọc thư viện.
Không dùng venv macOS đã upload. Không thay environment giữa dev và test.

Nếu chạy máy có Docker thay vì process executor:

```bash
docker build -t symplan-executor:v1 -f paperbench/Dockerfile .
export PAPER_EXECUTOR=docker
```

Bootstrap mặc định dành cho process executor; Docker vẫn cần cài model runtime và
dữ liệu nhưng không cần hạ quyền UID trong host process. Không đổi executor giữa suite.

## Chạy toàn bộ và resume

```bash
mkdir -p paper_runs
nohup bash scripts/run_paper_server.sh > paper_runs/paper_v3.log 2>&1 < /dev/null &
```

Mặc định chạy đủ **controlled + source + 3 ablations**. Mỗi suite tự chạy smoke 8
bài dev, dev đầy đủ cả hai dataset, freeze protocol rồi full test cả hai dataset.
Không có gate buộc SymPlan thắng mới chạy test. Cấu hình đã cố định; không tuning
trên test. Thất bại hạ tầng hoặc grading dừng run để sửa, không tính thành model sai.

```bash
tail -f paper_runs/paper_v3.log
cat paper_runs/paper_v3/status.json
```

Chạy lại đúng lệnh để resume. Mỗi record checkpoint theo `(problem_id, method)`.
Supervisor có file lock tránh chạy trùng. Thay code, prompt, model revision, runtime,
precision hoặc dữ liệu thì phải dùng thư mục mới và chạy lại dev. Không sửa/xóa
checkpoint để bỏ các bài sai. Script không tự tắt instance thuê sau khi hoàn tất.

Nếu chủ động chỉ chạy hai bảng trước (thiếu ablations cho kết luận thành phần):

```bash
PAPER_RUN_ROOT=paper_runs/two_tables bash scripts/run_paper_server.sh --suites controlled source
```

Không đổi danh sách suites khi resume cùng thư mục. Toàn bộ suite mặc định có
22.209 problem-method evaluations, gồm dev, smoke và test; đây không phải số giờ.
Lấy thời gian smoke thực tế để ước lượng chi phí, vì SymPlan có thêm calls.

## Dữ liệu và tính hợp lệ

`python -m paperbench.prepare` tạo manifest có commit SHA và checksum:
MATH-500 test 500 bài, GSM8K test 1.319 bài; mỗi dev 96 bài từ train, seed 42.
Không bỏ bài test vì khó hoặc timeout. Không dùng gold để sửa/chọn chương trình;
chỉ chấm sau solve, output thành công thì dừng ngay cả khi sai đáp án.
Lịch sử repo đã xem MATH-500 cần được công bố; không gọi đây là benchmark hoàn toàn
chưa từng dùng trong phát triển. Nếu muốn tuyên bố tổng quát hơn cần thêm model/data.

Freeze khóa model/revision, versions thư viện, Python/CUDA/GPU, source hash (gồm
prompt PaL vendored), dữ liệu và mọi cờ protocol. Các bảng source/controlled/ablation
được khóa riêng. Test không thay đổi lựa chọn của các suite tiếp theo.

## Lấy số đưa vào paper

Sau khi status là `complete`, lấy tại `paper_runs/paper_v3/tables/`:

- `metrics.csv`: đúng/N, accuracy đầu/cuối, ESR đầu/cuối, input/output tokens, calls, thời gian.
- `live_accuracy.json`: cập nhật sau từng method và sau mỗi câu hoàn tất; có dataset, model và precision.
- `paired.csv`: SymPlan trừ từng baseline, CI bootstrap 95%, McNemar exact và Holm p.
- `ablations.csv`: full trừ mỗi ablation, CI và p chưa hiệu chỉnh.
- `results.tex`: bảng LaTeX từ số thật, không điền số giả định hoặc tự bôi đậm winner.
- `provenance.json`: cấu hình và revision để truy vết.

Exporter từ chối run thiếu bài, run không phải test, protocol lệch hoặc grader error.
Mỗi thư mục run còn có `manifest.json`, `records.jsonl`, `summary.json`, `report.md`.
Có thể xuất lại bằng `python -m paperbench.export paper_runs/paper_v3` với code đã khóa.

Cách mô tả phù hợp: “We evaluate controlled program-aided adaptations of PaL, PoT,
and SymCode with a fixed Qwen3-4B-Instruct-2507 backbone, and report a separate
source-aligned prompt sensitivity study.” Không gọi đây là tái lập điểm số gốc.
Chỉ khẳng định cải thiện/significance khi số và kiểm định tương ứng hỗ trợ.

Nguồn: [model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/),
[Math-Verify](https://github.com/huggingface/Math-Verify).
