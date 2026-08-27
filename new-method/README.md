# new-method — Verifiable SymPlanner IR

Đây là nhánh thử nghiệm độc lập cho SymPlanner. Các bản cũ không bị sửa; mã tham chiếu được lưu trong `legacy/`:

- `legacy/current_method/`: extractor, planner prompt, sandbox và verifier hiện tại.
- `legacy/task2/`: extraction/module2, chuẩn hóa đơn vị/module3, codegen/module4, executor, few-shot/RAG và verify.
- `legacy/task2-tay/`: các pipeline extract/thinking hiện có.

## Điểm nâng cấp

Pipeline mới dùng hợp đồng dữ liệu cố định:

`question → ProblemIR → IR repair → normalized IR → codegen payload → sandbox execution → bidirectional relation verifier → targeted repair`

`ProblemIR` bắt buộc có `target_unknown`, `givens`, `relations`, `conditions`, `required_output`. Relation được canonicalize về expression dùng symbol ASCII, có `kind`, `operator`, `symbols`, `evidence`, `confidence`; payload có bảng chuyển đổi đơn vị. Codegen chỉ thấy payload đã chuẩn hóa, không tự đọc lại prose. Verifier kiểm tra cả chiều thuận (giá trị có thỏa quan hệ không) và chiều ngược (từ quan hệ có suy ra được biến/giá trị kỳ vọng không), sau đó phát diagnostic có cấu trúc theo từng relation/execution error cho repair.

Pipeline chạy theo nguyên tắc fail-closed: IR thiếu relation/metadata, symbol chưa khai báo, unit không hỗ trợ hoặc output code sai schema sẽ không được coi là hợp lệ. `source` và `evidence` chỉ dùng để audit extractor và bị loại khỏi computational payload. Intermediate variable chỉ được khai báo bởi vế trái dạng symbol của relation `kind="definition"` và phải xuất hiện trong `variables` nếu verifier cần nó để đánh giá graph.

## Chạy thử tối thiểu

```python
from new_method.pipeline import SymPlannerIRPipeline

pipeline = SymPlannerIRPipeline(llm_call=my_llm, execute_code=my_sandbox)
result = pipeline.run(question)
```

`my_llm(messages)` trả về text model; `my_sandbox(code)` trả về dict tối thiểu `{"answer": ..., "variables": {...}}`.

## Lộ trình tích hợp

1. Nối `llm_call` với adapter trong `legacy/task2/llm.py` hoặc model hiện tại.
2. Nối `execute_code` với sandbox hiện tại, giữ output JSON bắt buộc.
3. Chạy ablation: baseline, IR, IR+verify, IR+verify+repair trên cùng seed/dataset.
