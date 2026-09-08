"""Single-GPU non-thinking generation, with complete token and timing accounting."""
import time

MODELS = {
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
}


class Runner:
    def __init__(self, model_id, revision, precision="bf16", input_limit=8192, seed=42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. Run this command on the RTX 3090 host.")
        set_seed(seed)
        self.torch = torch
        self.input_limit = input_limit
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs = dict(revision=revision, torch_dtype=torch.bfloat16,
                      attn_implementation="sdpa", device_map={"": 0})
        if precision == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, messages, limit):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        # A chat template already inserts its own special tokens.
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count = inputs.input_ids.shape[-1]
        if count > self.input_limit:
            raise ValueError(f"Input has {count} tokens > {self.input_limit}; no silent truncation")
        inputs = inputs.to(self.model.device)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        self.torch.cuda.synchronize()
        tokens = output[0, count:]
        result = {"text": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                  "input_tokens": count, "output_tokens": len(tokens),
                  "limit_hit": len(tokens) >= limit, "seconds": time.monotonic() - start,
                  "peak_vram_gb": self.torch.cuda.max_memory_allocated() / 2**30}
        del inputs, output, tokens
        return result
