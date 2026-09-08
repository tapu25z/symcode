# Review SymPlan và kế hoạch thực nghiệm 4B

Review ngày 2026-09-08; bản thảo được tìm thấy ở `../latex-paper/main.tex`.

## Những điểm cần sửa trong paper

1. **Kết quả chưa được chứng minh.** Bảng Qwen3, Llama, các breakdown SymPlan và ablation có comment `Provisional`, nhưng abstract, discussion và conclusion khẳng định thắng trên mọi backbone. Không dùng các số này như kết quả đo. Thay bằng TBD hoặc bỏ bảng/claim cho đến khi có JSON đầy đủ; kiểm tra cả đồ thị ESR. Review này không thay số liệu trong LaTeX.
2. **Model không khớp.** Preset `llama3-8b` thực tế là `meta-llama/Meta-Llama-3-8B-Instruct`; paper ghi Llama-3.1. Qwen2.5 cần ghi rõ Coder-7B-Instruct, không chỉ Qwen2.5-7B. Kiểm tra citation đang gắn QwenMath cho model Coder và nguồn gốc MATH-500.
3. **Mẫu số có dấu hiệu khác nhau.** Accuracy 59.72% không thể là số câu đúng / 500 làm tròn hai chữ số. Cần báo correct/total và cùng tập câu cho mọi method; không loại execution failures khỏi accuracy.
4. **Đóng góp cần ablation mạnh hơn.** Extract -> Plan -> Code là giả thuyết hợp lý nhưng phải phân biệt với Plan-and-Solve/PAL/PoT bằng bằng chứng: giảm lỗi formulation nào, bao nhiêu, với chi phí bao nhiêu. So full với ExtractOnly, PlanOnly và NoModules trên cùng câu, cùng codegen prompt. SymCode hiện bị yêu cầu hai thuật toán trong khi SymPlan không bị yêu cầu này; đây là yếu tố gây nhiễu nếu chỉ so hai method.
5. **Budget không bằng nhau.** 1024 token mỗi call không phải cùng compute: SymPlan có nhiều call và retry. Báo tổng generated tokens, số call, latency, first-attempt accuracy và accuracy sau repair. Thêm so sánh CoT với tổng budget tương đương; không giảm budget baseline để tạo thứ hạng.
6. **Verifier không chứng minh tính đúng.** Code có heuristic theo dạng bài và có thể false pass/false fail. Cần ablation verifier và kiểm tra provenance của các luật nếu hình thành từ test set. Các ví dụ reconstruction chỉ minh họa cơ chế, không phải log thực nghiệm SymPlan.
7. **Paper/code cần đồng bộ.** Code retry cả khi verifier báo sai, không chỉ khi có traceback. Sau thay đổi này, codegen được phép sửa kế hoạch mâu thuẫn với đề; không nên mô tả nó như bắt buộc giữ nguyên formulation đúng.

## Chẩn đoán Qwen3

Preset Qwen3-8B đã tắt thinking mặc định, nên không thể kết luận thinking gây giảm accuracy ở các run dùng preset. Tuy nhiên, model-id trực tiếp hoặc `--enable-thinking` trước đây cho phép extract/plan dùng thinking với budget chỉ 192 token, trong khi codegen tắt thinking. Đây là cấu hình cần audit từ log thực tế.

Các rủi ro nhìn thấy trực tiếp trong code cũ: extract/plan 192 token; hai lần cắt note ở 1500 ký tự; tokenizer cắt prompt âm thầm ở 2560 token; prompt coi plan là chỉ dẫn phải theo; checkpoint ghi đè config rồi skip câu cũ. Chưa tìm thấy log Qwen3 có model ID trong JSON ở `result/`, nên đây là chẩn đoán code, chưa phải phân tích lỗi thực nghiệm Qwen3.

## Model và thay đổi v2

Ưu tiên **Qwen/Qwen3-4B-Instruct-2507**, model dense 4.0B chỉ có non-thinking, hỗ trợ toán/code và instruction following. Có thể dùng **Qwen/Qwen3-4B**, tắt thinking, làm đối chứng gần với Qwen3-8B. Lựa chọn dựa trên độ phù hợp pipeline, không có bằng chứng SymPlan sẽ thắng trên model này.

Nguồn: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 và https://huggingface.co/Qwen/Qwen3-4B . Transformers cần >=4.51.0 theo model card.

V2 dùng extract 384 / plan 768 token (CLI tùy chỉnh); tắt native thinking rõ ràng ở các stage này; plan phải nêu phương trình, thuật toán hữu hạn và kiểm tra miền; codegen đối chiếu lại đề. Bỏ cắt note theo ký tự, tăng context CLI lên 8192 và báo lỗi khi vượt giới hạn thay vì cắt đề/plan. Log token từng stage và cờ chạm budget; cờ này là tín hiệu cần xem raw output, không khẳng định chắc chắn output bị cắt. Checkpoint khác config hoặc không đọc được bị từ chối trước khi load model.

## Quy trình kiểm chứng

- Dùng tập development riêng (ví dụ GSM8K train), không chỉnh prompt theo đáp án MATH-500/GSM8K test rồi báo test như held-out.
- Smoke vài câu chỉ kiểm tra vận hành. Chốt cấu hình trên development rồi chạy toàn bộ test, giữ cả trường hợp SymPlan thua.
- Chạy 4 method chính và các ablation trên cùng checkpoint mới của từng model. Dùng BF16 trên GPU nếu phù hợp; nếu NF4, áp dụng cho mọi method và ghi rõ.
- So kết quả paired theo câu: CoT đúng/SymPlan sai và chiều ngược lại. Phân loại lỗi extraction, formulation, plan, code, timeout, output parser. Báo khoảng tin cậy paired hoặc McNemar khi so accuracy.
- Nếu 4B vẫn ưu tiên CoT, báo đó là giới hạn phụ thuộc backbone. Không có sửa code nào đảm bảo thứ hạng trước khi đo.
