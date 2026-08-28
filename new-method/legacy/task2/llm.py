#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared HuggingFace LLM runtime for the Task 2 pipeline.

This remake intentionally loads the base Qwen2.5-7B-Instruct model only.
LoRA support remains available only when explicit adapter paths are passed.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional


MODULE2_ADAPTER_NAME = "module2"
MODULE4_ADAPTER_NAME = "module4"

DEFAULT_MODULE2_ADAPTER_PATH = (
    r"D:\TAYBUOI_EXACT2026\XAI2026\task2\fine_tune_extract\qwen2.5_lora_module2"
)

DEFAULT_MODULE4_ADAPTER_PATH = (
    r"D:\EXACT2026\logicllama\LogicLLaMA\XAI2026\task2\fine_tune_simpycode\qwen2.5_lora_sympy_new\qwen2.5_lora_sympy_new"
)

ADAPTER_ALIASES: Dict[str, str] = {
    "module2": MODULE2_ADAPTER_NAME,
    "module_2": MODULE2_ADAPTER_NAME,
    "m2": MODULE2_ADAPTER_NAME,
    "extract": MODULE2_ADAPTER_NAME,

    "module4": MODULE4_ADAPTER_NAME,
    "module_4": MODULE4_ADAPTER_NAME,
    "m4": MODULE4_ADAPTER_NAME,
    "sympy": MODULE4_ADAPTER_NAME,
    "symcode": MODULE4_ADAPTER_NAME,
}


def _import_hf_runtime() -> Dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "HuggingFace runtime dependencies are not ready. "
            "Run `python task2/scripts/setup_ollama.py` or "
            "`pip install -r task2/requirements.txt` first. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


class RemoteVLLMClient:
    """Client for calling vLLM/OpenAI compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        use_adapter: Any = False,
        module_id: Any = None,
    ) -> str:
        # Task 2 Physics doesn't use adapters when shared with Type 1 vLLM
        # as per user instructions.
        del use_adapter, module_id

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={"repetition_penalty": self.repetition_penalty},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  vLLM API Error: {e}", flush=True)
            return ""


class LocalHFLoraClient:
    """Small chat client with module-specific LoRA adapter routing."""

    def __init__(
        self,
        model_name: str,
        adapters: Optional[Dict[str, str]] = None,
        *,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_compute_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        max_input_tokens: int = 4096,
        tokenizer_padding_side: str = "right",
    ) -> None:
        self.model_name = model_name
        self.adapter_paths: Dict[str, str] = {}

        if adapters:
            for name, path in adapters.items():
                clean_name = str(name or "").strip()
                clean_path = str(path or "").strip()
                if clean_name and clean_path:
                    self.adapter_paths[clean_name] = str(Path(clean_path).expanduser())

        self._warned_missing_adapters: set[str] = set()
        self.bnb_compute_dtype = bnb_compute_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_input_tokens = max_input_tokens
        self.tokenizer_padding_side = tokenizer_padding_side

        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose only one quantization mode: 4-bit or 8-bit.")

        runtime = _import_hf_runtime()
        self.torch = runtime["torch"]
        self.PeftModel = runtime["PeftModel"]
        self.AutoModelForCausalLM = runtime["AutoModelForCausalLM"]
        self.AutoTokenizer = runtime["AutoTokenizer"]
        self.BitsAndBytesConfig = runtime["BitsAndBytesConfig"]

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        self.model.eval()

    def _load_tokenizer(self) -> Any:
        tokenizer = self.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = self.tokenizer_padding_side
        tokenizer.truncation_side = "left"
        return tokenizer

    def _resolve_compute_dtype(self) -> Any:
        requested = self.bnb_compute_dtype.lower().strip()

        if requested == "auto":
            is_bf16_supported = getattr(self.torch.cuda, "is_bf16_supported", None)
            if (
                self.torch.cuda.is_available()
                and callable(is_bf16_supported)
                and is_bf16_supported()
            ):
                return self.torch.bfloat16
            return self.torch.float16

        if requested in {"bf16", "bfloat16"}:
            return self.torch.bfloat16

        if requested in {"fp16", "float16"}:
            return self.torch.float16

        if requested in {"fp32", "float32"}:
            return self.torch.float32

        raise ValueError(
            "Invalid bnb_compute_dtype. Use one of: auto, float16, bfloat16, float32."
        )

    def _quantization_config(
        self,
        *,
        load_in_4bit: bool,
        load_in_8bit: bool,
    ) -> Optional[Any]:
        if load_in_4bit:
            compute_dtype = self._resolve_compute_dtype()
            return self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

        if load_in_8bit:
            return self.BitsAndBytesConfig(load_in_8bit=True)

        return None

    def _max_memory(self) -> Optional[Dict[Any, str]]:
        if not self.torch.cuda.is_available():
            return None

        override = os.getenv("HF_MAX_MEMORY_PER_GPU", "").strip()
        if override:
            per_gpu = override
        else:
            total_gib = self.torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            per_gpu = f"{max(int(total_gib) - 1, 1)}GiB"

        max_memory: Dict[Any, str] = {
            gpu_id: per_gpu for gpu_id in range(self.torch.cuda.device_count())
        }

        cpu_memory = os.getenv("HF_MAX_CPU_MEMORY", "").strip()
        if cpu_memory:
            max_memory["cpu"] = cpu_memory

        return max_memory

    def _load_model(
        self,
        *,
        load_in_4bit: bool,
        load_in_8bit: bool,
    ) -> Any:
        quantization_config = self._quantization_config(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        device_map: Any = "auto" if self.torch.cuda.is_available() else None
        dtype = self.torch.bfloat16 if self.torch.cuda.is_available() else self.torch.float32
        attn_implementation = os.getenv("HF_ATTN_IMPLEMENTATION", "sdpa").strip() or "sdpa"

        print(f"  HF: loading base model: {self.model_name}", flush=True)
        print(
            "  HF: device_map=%s, dtype=%s, quantization=%s, bnb_compute_dtype=%s"
            % (
                device_map,
                str(dtype).replace("torch.", ""),
                "4-bit NF4" if load_in_4bit else "8-bit" if load_in_8bit else "none",
                str(self._resolve_compute_dtype()).replace("torch.", "")
                if load_in_4bit
                else "n/a",
            ),
            flush=True,
        )

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        max_memory = self._max_memory()
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            try:
                base_model = self.AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
            except TypeError:
                model_kwargs.pop("attn_implementation", None)
                if "dtype" in model_kwargs:
                    model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")

                base_model = self.AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )

        except Exception as exc:
            if quantization_config is not None:
                mode = "4-bit NF4" if load_in_4bit else "8-bit"
                raise RuntimeError(
                    f"Failed to load the quantized HF model in {mode} mode. "
                    "Check that CUDA, bitsandbytes, torch, transformers, peft, "
                    "and accelerate match your runtime. For inference.py use "
                    "`--load-mode fp16` to run without quantization, or "
                    "`--load-mode 8bit` if 4-bit is unavailable. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            raise

        if not self.adapter_paths:
            print("  HF: no LoRA adapter path configured; using base model only", flush=True)
            return base_model

        adapter_items = list(self.adapter_paths.items())

        first_name, first_path_text = adapter_items[0]
        first_path = Path(first_path_text)

        if not first_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter path not found for '{first_name}': {first_path}"
            )

        print(f"  HF: loading LoRA adapter '{first_name}': {first_path}", flush=True)

        model = self.PeftModel.from_pretrained(
            base_model,
            str(first_path),
            adapter_name=first_name,
            is_trainable=False,
        )

        for adapter_name, adapter_path_text in adapter_items[1:]:
            adapter_path = Path(adapter_path_text)

            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"LoRA adapter path not found for '{adapter_name}': {adapter_path}"
                )

            print(f"  HF: loading LoRA adapter '{adapter_name}': {adapter_path}", flush=True)

            model.load_adapter(
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )

        print(
            "  HF: loaded LoRA adapters: %s"
            % ", ".join(self.adapter_paths.keys()),
            flush=True,
        )

        return model

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            chunks: List[str] = []

            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                chunks.append(f"{role}:\n{content}")

            chunks.append("assistant:\n")
            return "\n\n".join(chunks)

    def _resolve_adapter_name(
        self,
        use_adapter: Any,
        *,
        module_id: Any = None,
    ) -> Optional[str]:
        """Resolve adapter for a single LLM call.

        Rules:
        - module_id=2 -> module2 adapter
        - module_id=4 -> module4 adapter
        - use_adapter=True -> module4 adapter (changed from module2 for Task 2 logic)
        - use_adapter="module2"/"extract" -> module2 adapter
        - use_adapter="module4"/"sympy"/"symcode" -> module4 adapter
        - use_adapter=False and module_id=None -> base model
        """

        if module_id is not None:
            module_text = str(module_id).strip().lower().replace(" ", "")

            if module_text in {"2", "module2", "module_2", "m2"}:
                return MODULE2_ADAPTER_NAME

            if module_text in {"4", "module4", "module_4", "m4"}:
                return MODULE4_ADAPTER_NAME

            return None

        if use_adapter is True:
            # Default 'True' in Task 2 context usually means the specialized solver adapter
            return MODULE4_ADAPTER_NAME

        if isinstance(use_adapter, str) and use_adapter.strip():
            requested = use_adapter.strip().lower()
            return ADAPTER_ALIASES.get(requested, requested)

        return None

    @contextmanager
    def _adapter_context(
        self,
        *,
        use_adapter: Any,
        module_id: Any = None,
    ) -> Any:
        adapter_name = self._resolve_adapter_name(
            use_adapter,
            module_id=module_id,
        )

        if adapter_name:
            if adapter_name not in self.adapter_paths:
                if adapter_name not in self._warned_missing_adapters:
                    print(
                        f"  HF: requested adapter '{adapter_name}' is not loaded; "
                        "using base model for this call",
                        flush=True,
                    )
                    self._warned_missing_adapters.add(adapter_name)

                disable_adapter = getattr(self.model, "disable_adapter", None)
                if callable(disable_adapter):
                    with disable_adapter():
                        yield
                    return

                with nullcontext():
                    yield
                return

            set_adapter = getattr(self.model, "set_adapter", None)

            if not callable(set_adapter):
                yield
                return

            active_adapter = getattr(self.model, "active_adapter", None)
            previous_adapter = active_adapter() if callable(active_adapter) else active_adapter

            set_adapter(adapter_name)

            try:
                yield
            finally:
                if previous_adapter:
                    try:
                        set_adapter(previous_adapter)
                    except Exception:
                        pass

            return

        disable_adapter = getattr(self.model, "disable_adapter", None)

        if callable(disable_adapter):
            with disable_adapter():
                yield
            return

        with nullcontext():
            yield

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        use_adapter: Any = False,
        module_id: Any = None,
    ) -> str:
        prompt = self._format_messages(messages)

        tokenizer_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
        }

        if self.max_input_tokens > 0:
            tokenizer_kwargs.update(
                {
                    "truncation": True,
                    "max_length": self.max_input_tokens,
                }
            )

        inputs = self.tokenizer(prompt, **tokenizer_kwargs)

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {
                key: value.to(model_device)
                for key, value in inputs.items()
            }

        do_sample = self.temperature > 0

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = do_sample

        if not do_sample:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "generation_config": generation_config,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }

        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with self._adapter_context(
            use_adapter=use_adapter,
            module_id=module_id,
        ):
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()


def build_client(
    host: str = "",
    api_key: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    extra_adapters: Optional[Dict[str, str]] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_compute_dtype: str = "auto",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    max_input_tokens: int = 4096,
    tokenizer_padding_side: str = "right",
) -> Any:
    """Build either a RemoteVLLMClient (if host is provided) or LocalHFLoraClient."""

    if host:
        print(f"  Task 2: Using remote vLLM API at {host}", flush=True)
        return RemoteVLLMClient(
            base_url=host,
            api_key=api_key or "EMPTY",
            model_name=model_name or "Qwen/Qwen3-8B",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    if not model_name:
        model_name = os.getenv("HF_BASE_MODEL", "Qwen/Qwen3-8B")

    # Final adapter mapping
    adapters: Dict[str, str] = {}

    # 1. Module 2 adapter is not auto-loaded in task2-remake.
    # 2. Module 4 adapter is opt-in only through an explicit adapter_path.
    m4_path = adapter_path
    if m4_path and Path(m4_path).exists():
        adapters[MODULE4_ADAPTER_NAME] = m4_path

    # 3. Apply any explicit extra_adapters
    if extra_adapters:
        for name, path in extra_adapters.items():
            if path and Path(path).exists():
                adapters[name] = path

    return LocalHFLoraClient(
        model_name=model_name,
        adapters=adapters,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_compute_dtype=bnb_compute_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_input_tokens=max_input_tokens,
        tokenizer_padding_side=tokenizer_padding_side,
    )


def chat_messages(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    *,
    use_adapter: Any = False,
    module_id: Any = None,
) -> str:
    """Send chat messages to the model and return assistant text."""

    del model

    return client.chat(
        messages,
        use_adapter=use_adapter,
        module_id=module_id,
    )
