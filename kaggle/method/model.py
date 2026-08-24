import os
from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Prevent CUDA memory fragmentation on long benchmark runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class LLMRunner:
    """
    Manages model initialization, 4-bit quantization, and chat-based inference
    with memory-efficient SDPA attention and strict VRAM deallocation.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device_map: str = "auto",
        hf_token: Optional[str] = None
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        token = hf_token or os.environ.get("HF_TOKEN") or None
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        print(f"[*] Initializing model: {model_id} (4-bit: {load_in_4bit}, dtype: {compute_dtype})...")
        
        bnb_config = None
        if load_in_4bit:
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

        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # Load with SDPA (Scaled Dot-Product Attention) for O(1) memory per token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            quantization_config=bnb_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            attn_implementation="sdpa"
        )
        self.model.eval()
        print(f"[*] Model successfully loaded on device: {self.model.device} with SDPA attention!")

    def generate_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """
        Runs generation for a ChatML message list with memory-safe tensor handling.
        Returns:
            (generated_text, generated_token_count)
        """
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback formatting if model does not have a chat_template
            formatted = []
            for m in messages:
                role = m.get("role", "user").capitalize()
                content = m.get("content", "")
                formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            formatted.append("<|im_start|>assistant\n")
            prompt = "\n".join(formatted)

        # Truncate prompt if context grows excessively large to prevent OOM
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2560
        ).to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
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

        # Explicitly release GPU tensor allocations to avoid memory buildup
        del inputs
        del outputs
        del gen_tokens

        return generated_text, num_generated_tokens
