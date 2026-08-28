# new-method — SymPlanner IR / Adaptive Computation Graph

Đây là nhánh thử nghiệm độc lập cho SymPlanner. Các bản cũ không bị sửa; mã tham chiếu được lưu trong `legacy/`:

- `legacy/current_method/`: extractor, planner prompt, sandbox và verifier hiện tại.
- `legacy/task2/`: extraction/module2, chuẩn hóa đơn vị/module3, codegen/module4, executor, few-shot/RAG và verify.
- `legacy/task2-tay/`: các pipeline extract/thinking hiện có.

## Hướng hiện tại

Pipeline tương thích hiện tại:

`question -> extract IR -> normalize IR -> plan/codegen -> sandbox -> bidirectional verify -> tối đa 2 code repair`

Extractor chỉ chạy một lần và không có IR repair. Normalizer điền default cho lỗi hình thức nhỏ; chỉ IR không có target hoặc quan hệ dùng được mới bị đánh dấu `invalid_ir`. Code vẫn in JSON một dòng gồm `answer`, `canonical_answer`, `answer_type`, `unit`, `variables` để evaluator/scorer đọc ổn định.

Bidirectional verifier kiểm tra output contract, quan hệ theo hai chiều và conditions. Mọi lỗi sau sandbox đều dùng chung code repair, tối đa hai lần.

### Novelty method: ACG-IR

`ACG` là biến thể mới của pipeline, dùng `Adaptive Computation Graph IR` thay cho taxonomy cố định theo GSM8K/MATH500:

`question -> graph extraction -> normalization -> structure-driven solver plan -> codegen -> execution -> relation verification`

Planner chọn chiến lược từ graph: `sequential_eval`, `symbolic_solve`, `enumerative_search`, `combinatorics_formula`, `geometry_helper` hoặc `hybrid`. Dataset chỉ còn là metadata đánh giá; không được dùng làm nhánh kiến trúc.

IR mới cũng coi unit chưa biết là nhãn opaque/ambiguous thay vì lỗi parse. Vì vậy các entity như `dogs`, `pages`, `packs`, `boxes` vẫn đi qua pipeline và được verifier kiểm tra theo quan hệ số học.

Chạy benchmark novelty method:

```powershell
python new-method/run_benchmark.py --dataset gsm8k --num-samples 100 --methods ACG
python new-method/run_benchmark.py --dataset math500 --num-samples 50 --methods ACG
```

Spec thiết kế và kế hoạch ablation nằm tại [`ADAPTIVE_COMPUTATION_GRAPH_PLAN.md`](./ADAPTIVE_COMPUTATION_GRAPH_PLAN.md).

## Chạy thử tối thiểu

```python
from new_method.pipeline import SymPlannerIRPipeline

pipeline = SymPlannerIRPipeline(llm_call=my_llm, execute_code=my_sandbox)
result = pipeline.run(question)
```

`my_llm(messages)` trả về text model; `my_sandbox(code)` trả về dict theo contract đầy đủ bên dưới.

Contract sandbox hiện tại là một JSON line gồm `answer`, `canonical_answer`, `answer_type`, `unit`, `variables`. Adapter trong `new_method/adapters.py` chuyển output của sandbox cũ sang contract này.

## Benchmark

Runner mới dùng cùng `Qwen/Qwen2.5-Coder-7B-Instruct`, dataset loader, exact-match và result schema với benchmark cũ:

```powershell
python new-method/run_benchmark.py --dataset math500 --num-samples 50 --methods SymPlanner IR
```

- `SymPlanner`: baseline cũ.
- `IR`: extractor + normalizer + plan/codegen + sandbox + bidirectional verifier + tối đa 2 code repair.

Import hoặc chạy `--help` không tải model. Model chỉ được khởi tạo khi `main()` bắt đầu một benchmark thật.

## Trạng thái tích hợp

Adapter model, sandbox, evaluator/checkpoint và runner benchmark dùng chung với baseline. Nên chạy smoke 10 mẫu với `SymPlanner IR` trước khi chạy full benchmark.
