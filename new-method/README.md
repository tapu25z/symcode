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

Math500 được hỗ trợ ở mức numeric, exact fraction, symbolic expression, tuple, finite set và interval. Output tách `answer` (chuỗi dùng để chấm với dataset) khỏi `canonical_answer` (biểu thức SymPy dùng cho verifier), tránh ép mọi bài về số thực.

## Chạy thử tối thiểu

```python
from new_method.pipeline import SymPlannerIRPipeline

pipeline = SymPlannerIRPipeline(llm_call=my_llm, execute_code=my_sandbox)
result = pipeline.run(question)
```

`my_llm(messages)` trả về text model; `my_sandbox(code)` trả về dict theo contract đầy đủ bên dưới.

Contract sandbox hiện tại là một JSON line gồm `answer`, `canonical_answer`, `answer_type`, `unit`, `variables`. Adapter trong `new_method/adapters.py` chuyển output của sandbox cũ sang contract này.

## Benchmark và ablation

Runner mới dùng cùng `Qwen/Qwen2.5-Coder-7B-Instruct`, dataset loader, exact-match và result schema với benchmark cũ:

```powershell
python new-method/run_benchmark.py --dataset math500 --num-samples 50 --methods SymPlanner IR-Codegen IR-BiVerify IR-Full
```

- `SymPlanner`: baseline cũ.
- `IR-Codegen`: structured IR + normalization + codegen, chỉ kiểm tra output contract.
- `IR-BiVerify`: thêm bidirectional verifier, không code repair.
- `IR-Full`: verifier và targeted repair đầy đủ.

Import hoặc chạy `--help` không tải model. Model chỉ được khởi tạo khi `main()` bắt đầu một benchmark thật.

## Trạng thái tích hợp

Adapter model, sandbox, evaluator/checkpoint và runner ablation đã có. Model thật chưa được chạy; trước full benchmark nên chạy smoke 3–5 mẫu cho từng dataset để xác nhận VRAM, context length và output compliance của model.
