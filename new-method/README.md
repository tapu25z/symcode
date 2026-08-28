# new-method — Lean SymPlanner

Đây là nhánh thử nghiệm độc lập cho SymPlanner. Các bản cũ không bị sửa; mã tham chiếu được lưu trong `legacy/`:

- `legacy/current_method/`: extractor, planner prompt, sandbox và verifier hiện tại.
- `legacy/task2/`: extraction/module2, chuẩn hóa đơn vị/module3, codegen/module4, executor, few-shot/RAG và verify.
- `legacy/task2-tay/`: các pipeline extract/thinking hiện có.

## Huong hien tai

Duong chay chinh da duoc cat gon de tranh over-engineering:

`question -> codegen JSON-contract Python -> sandbox execution -> lightweight verifier -> targeted repair`

`IR-Lite` va `IR-Full` hien la cung mot duong lean problem-to-code. Khong con buoc extract/repair IR bat buoc, nen loi schema khong lam ca mau thanh `invalid_ir` truoc khi sinh code. Code van in JSON mot dong gom `answer`, `canonical_answer`, `answer_type`, `unit`, `variables` de evaluator/scorer doc on dinh.

Verifier trong duong lean chi la guard nhe: runtime error, output JSON sai contract, `None`/`NaN`/vo cuc, free symbols ro rang, va mot so rang buoc mien gia tri doc lap. No khong co gang chung minh lai toan bo relation graph.

Duong strict IR van duoc giu de nghien cuu/ablation duoi ten `IR-Strict`. Cac bien the `IR-Codegen` va `IR-BiVerify` van dung structured IR nhu truoc.

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
python new-method/run_benchmark.py --dataset math500 --num-samples 50 --methods SymPlanner IR-Lite
```

- `SymPlanner`: baseline cũ.
- `IR-Lite`: duong lean mac dinh, 1 codegen + toi da 1 repair theo default.
- `IR-Full`: alias tuong thich nguoc cua `IR-Lite`, dung khi script cu da goi ten nay.
- `IR-Codegen`: structured IR + normalization + codegen, chi kiem tra output contract.
- `IR-BiVerify`: structured IR + bidirectional verifier, khong code repair.
- `IR-Strict`: structured IR + bidirectional verifier + targeted repair day du nhu thiet ke cu.

Import hoặc chạy `--help` không tải model. Model chỉ được khởi tạo khi `main()` bắt đầu một benchmark thật.

## Trạng thái tích hợp

Adapter model, sandbox, evaluator/checkpoint va runner ablation da co. Truoc full benchmark nen chay smoke 10 mau voi `IR-Lite` de so truc tiep voi baseline, sau do moi quyet dinh co can bat lai `IR-Strict` cho nhom bai nao hay khong.
