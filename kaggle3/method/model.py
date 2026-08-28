"""
Module quản lý mô hình ngôn ngữ lớn (LLMRunner) cho benchmark:
Sử dụng chuẩn Hugging Face Transformers với lượng tử hóa 4-bit BitsAndBytes NF4 (load_in_4bit=True).
Tối ưu bộ nhớ với PyTorch SDPA (Scaled Dot-Product Attention) và giải phóng VRAM an toàn.
"""

import os
import gc
from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Thiết lập quản lý phân mảnh bộ nhớ GPU cho các phiên chạy benchmark dài (trên Linux)
if os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class LLMRunner:
    """
    Quản lý khởi tạo mô hình Hugging Face Transformers với lượng tử hóa 4-bit BitsAndBytes NF4 chuẩn hóa.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        use_vllm: bool = False,
        load_in_4bit: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_input_tokens: int = 2560,
        device_map: str = "cuda:0",
        hf_token: Optional[str] = None
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_input_tokens = int(max_input_tokens)
        self.use_vllm = use_vllm
        self.load_in_4bit = load_in_4bit
        self.is_vllm_active = False

        token = hf_token or os.environ.get("HF_TOKEN") or None
        cuda_avail = torch.cuda.is_available()

        if cuda_avail:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            target_device = "cuda:0"
        else:
            compute_dtype = torch.float32
            self.load_in_4bit = False
            target_device = "cpu"
            self.use_vllm = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Khởi tạo vLLM nếu explicitly bật (chế độ tùy chọn)
        if self.use_vllm and cuda_avail:
            try:
                from vllm import LLM
                vllm_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" if (self.load_in_4bit and model_id == "Qwen/Qwen2.5-Coder-7B-Instruct") else model_id
                print(f"[INFO] Khởi tạo vLLM Engine (tùy chọn): {vllm_model_id}")
                self.vllm_engine = LLM(
                    model=vllm_model_id,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.80,
                    max_model_len=self.max_input_tokens + self.max_new_tokens,
                    trust_remote_code=True
                )
                self.is_vllm_active = True
                print(f"[INFO] vLLM Engine khởi tạo thành công.")
            except Exception as err:
                print(f"[WARN] vLLM không khởi tạo được ({err}). Chuyển sang Hugging Face 4-bit chuẩn.")
                self.is_vllm_active = False

        # Khởi tạo chuẩn mặc định bằng Hugging Face Transformers 4-bit BitsAndBytes
        if not self.is_vllm_active:
            print(f"[INFO] Khởi tạo HF Transformers: {model_id} (load_in_4bit: {self.load_in_4bit}, dtype: {compute_dtype}, device: {target_device})")

            bnb_config = None
            if self.load_in_4bit and cuda_avail:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True
                )

            if cuda_avail:
                gc.collect()
                torch.cuda.empty_cache()

            model_kwargs = {
                "token": token,
                "device_map": target_device,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "dtype": compute_dtype
            }

            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config

            if cuda_avail:
                model_kwargs["attn_implementation"] = "sdpa"

            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            except Exception as err:
                if "attn_implementation" in model_kwargs:
                    del model_kwargs["attn_implementation"]
                    if cuda_avail:
                        gc.collect()
                        torch.cuda.empty_cache()
                    self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                else:
                    raise err

            self.model.eval()
            print(f"[INFO] Mô hình HF Transformers 4-bit đã được nạp thành công trên thiết bị: {self.model.device}")

    def _format_chat_prompt(self, messages: List[Dict[str, str]], enable_thinking: Optional[bool] = None) -> str:
        """Định dạng danh sách tin nhắn ChatML thành chuỗi prompt."""
        chat_template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True
        }
        if enable_thinking is not None:
            try:
                chat_template_kwargs["enable_thinking"] = enable_thinking
            except TypeError:
                pass

        try:
            return self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        except (TypeError, Exception):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted = []
                for m in messages:
                    role = m.get("role", "user").capitalize()
                    content = m.get("content", "")
                    formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                formatted.append("<|im_start|>assistant\n")
                return "\n".join(formatted)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens_override: Optional[int] = None,
        enable_thinking: Optional[bool] = None
    ) -> Tuple[str, int]:
        """
        Sinh câu trả lời từ danh sách thông điệp ChatML qua HF Transformers.
        Returns: (generated_text, generated_token_count)
        """
        prompt = self._format_chat_prompt(messages, enable_thinking=enable_thinking)
        tokens_limit = int(max_new_tokens_override) if max_new_tokens_override is not None else self.max_new_tokens

        # Nếu đang dùng vLLM Engine
        if self.is_vllm_active:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=self.temperature if self.temperature > 0.0 else 0.0,
                top_p=self.top_p if self.temperature > 0.0 else 1.0,
                max_tokens=tokens_limit
            )
            outputs = self.vllm_engine.generate([prompt], sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0]
            return output.text.strip(), len(output.token_ids)

        # MẶC ĐỊNH: HUGGING FACE TRANSFORMERS 4-BIT BITSANDBYTES
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        ).to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=tokens_limit,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                top_p=self.top_p if self.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        gen_tokens = outputs[0][input_len:]
        num_generated_tokens = len(gen_tokens)
        generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        del inputs
        del outputs
        del gen_tokens

        return generated_text, num_generated_tokens

    def generate_chat_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens_override: Optional[int] = None,
        enable_thinking: Optional[bool] = None
    ) -> List[Tuple[str, int]]:
        """
        Sinh câu trả lời cho danh sách nhiều prompt dạng batch.
        Returns: List of (generated_text, generated_token_count)
        """
        results = []
        for msgs in messages_list:
            results.append(self.generate_chat(msgs, max_new_tokens_override=max_new_tokens_override, enable_thinking=enable_thinking))
        return results
