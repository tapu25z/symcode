"""
Module quản lý mô hình ngôn ngữ lớn (LLMRunner) với cơ chế lượng tử hóa 4-bit (bitsandbytes),
tối ưu bộ nhớ qua cơ chế SDPA (Scaled Dot-Product Attention) và giải phóng VRAM an toàn.
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
    Quản lý khởi tạo mô hình, cấu hình lượng tử hóa và thực hiện sinh văn bản theo cấu trúc ChatML.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_input_tokens: int = 2560,
        device_map: str = "auto",
        hf_token: Optional[str] = None
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_input_tokens = int(max_input_tokens)
        
        token = hf_token or os.environ.get("HF_TOKEN") or None
        cuda_avail = torch.cuda.is_available()
        
        if cuda_avail:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            compute_dtype = torch.float32
            load_in_4bit = False
            device_map = "cpu"
        
        print(f"[INFO] Khoi tao mo hinh: {model_id} (4-bit: {load_in_4bit}, dtype: {compute_dtype}, device: {device_map})")
        
        bnb_config = None
        if load_in_4bit and cuda_avail:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if cuda_avail:
            gc.collect()
            torch.cuda.empty_cache()

        model_kwargs = {
            "token": token,
            "device_map": device_map,
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
            # Fallback nếu SDPA không tương thích với một số phiên bản PyTorch cũ
            if "attn_implementation" in model_kwargs:
                del model_kwargs["attn_implementation"]
                if cuda_avail:
                    gc.collect()
                    torch.cuda.empty_cache()
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            else:
                raise err

        self.model.eval()
        print(f"[INFO] Mo hinh da duoc tai thanh cong tren thiet bi: {self.model.device}")

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens_override: Optional[int] = None,
        enable_thinking: Optional[bool] = None
    ) -> Tuple[str, int]:
        """
        Sinh câu trả lời từ danh sách thông điệp ChatML với cơ chế thu hồi bộ nhớ tensor nghiêm ngặt.
        
        Returns:
            (generated_text, generated_token_count)
        """
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
            prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        except (TypeError, Exception):
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted = []
                for m in messages:
                    role = m.get("role", "user").capitalize()
                    content = m.get("content", "")
                    formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                formatted.append("<|im_start|>assistant\n")
                prompt = "\n".join(formatted)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        ).to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        tokens_limit = int(max_new_tokens_override) if max_new_tokens_override is not None else self.max_new_tokens

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

        # Giải phóng biến tensor để tránh tích lũy bộ nhớ GPU
        del inputs
        del outputs
        del gen_tokens

        return generated_text, num_generated_tokens

