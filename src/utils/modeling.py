"""Transformers helpers with CPU fallbacks for local smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.common import model_dtype, resolve_device


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str | Path, *, quantization_config: Any = None, training: bool = False):
    import torch
    from transformers import AutoModelForCausalLM
    kwargs: dict[str, Any] = {"torch_dtype": model_dtype()}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    elif resolve_device() == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
    if training and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    elif not training and resolve_device() == "cpu":
        model.to(torch.device("cpu"))
    return model


def generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 100, do_sample: bool = False, temperature: float = 0.7) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.pad_token_id}
    if do_sample:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

