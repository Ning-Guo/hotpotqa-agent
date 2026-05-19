# src/models.py
"""
Model loading for the live inference agent.

Loads a single PeftModel instance (base + GRPO LoRA adapter). The same model
serves two roles:

  Reasoning mode  (classify, decompose, rewrite, answer_hop1):
      with model.disable_adapter(): model.generate(...)
      → base Qwen2.5-3B-Instruct, no fine-tuning bias

  Answer mode (final answer synthesis):
      model.generate(...)
      → GRPO-tuned LoRA adapter active

Loading one model instead of two halves peak memory usage.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    adapter_repo: str,
    device=None,
) -> tuple:
    """
    Returns (model, tokenizer, device).

    model_name   : HuggingFace base model ID (e.g. "Qwen/Qwen2.5-3B-Instruct")
    adapter_repo : HF repo ID or local path to the GRPO LoRA adapter
    device       : "cuda" | "mps" | "cpu"  (auto-detected if None)
    """
    device = device or detect_device()
    print(f"Loading '{model_name}' + adapter '{adapter_repo}' on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if device == "cuda":
    #     from transformers import BitsAndBytesConfig
    #     bnb_cfg = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #     )
    #     base = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         quantization_config=bnb_cfg,
    #         device_map="auto",
    #     )
    # else:
    #     base = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16,
    #     ).to(device)

    if device == "cuda":
        # Load in bfloat16 — no quantization needed on A100/H100 (80 GB VRAM).
        # The 3B model uses ~6 GB, well within budget.
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)

    model = PeftModel.from_pretrained(base, adapter_repo)
    model.eval()

    print("Model ready.")
    return model, tokenizer, device
