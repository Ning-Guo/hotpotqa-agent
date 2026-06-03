# training/utils/inference.py
"""
Teacher model inference backends and prompt templates.
Shared by 02_generate_sft_data.py and 02b_generate_eval_traces.py.
"""

from __future__ import annotations

import time


# ---------------------------------------------------------------------------
# Prompt templates (work-backwards from gold answer)
# ---------------------------------------------------------------------------

BRIDGE_TEMPLATE = """\
You are a multi-hop question answering expert.

Given the question, passages, and the CORRECT final answer, generate the \
step-by-step reasoning chain that leads to that answer using ONLY the \
provided passages. Work backwards from the answer.

Question: {question}
Question type: bridge (requires finding an intermediate entity first)
Correct answer: {answer}

Passages:
{context}

Generate your response in EXACTLY this format — no other text:

<think>
Type: bridge
Sub-question 1: [first sub-question to find the intermediate entity]
Intermediate answer: [answer to sub-question 1, must appear verbatim in passages]
Sub-question 2: [second sub-question using the intermediate entity]
</think>
<answer>{answer}</answer>

Rules:
- Sub-question 1 MUST identify the intermediate entity, NOT restate the original question
- Intermediate answer MUST appear word-for-word in one of the passages above
- Sub-question 2 MUST use the intermediate answer to ask for the final answer
- The <answer> tag MUST contain exactly: {answer}
"""

COMPARISON_TEMPLATE = """\
You are a multi-hop question answering expert.

Given the question, passages, and the CORRECT final answer, generate the \
step-by-step reasoning chain that leads to that answer using ONLY the \
provided passages. Work backwards from the answer.

Question: {question}
Question type: comparison (compare a property of two entities)
Correct answer: {answer}

Passages:
{context}

Generate your response in EXACTLY this format — no other text:

<think>
Type: comparison
Sub-question 1: [question about the property of entity 1]
Intermediate answer 1: [answer about entity 1, must appear in passages]
Sub-question 2: [question about the property of entity 2]
Intermediate answer 2: [answer about entity 2, must appear in passages]
</think>
<answer>{answer}</answer>

Rules:
- Each intermediate answer MUST appear word-for-word in one of the passages above
- The <answer> tag MUST contain exactly: {answer}
"""

INSUFFICIENT_TEMPLATE = """\
You are a multi-hop question answering expert.

Given the question and passages, the supporting passages needed to answer \
this question are NOT present. Generate a response acknowledging this.

Question: {question}

Passages:
{context}

Generate your response in EXACTLY this format:

<think>
Type: {qtype}
Note: The supporting passages for this question are not available in the context.
</think>
<answer>insufficient context</answer>
"""


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def run_api_inference(prompts: list[str], model: str, base_url: str,
                      api_key: str, max_tokens: int = 512) -> list[str]:
    """OpenAI-compatible API inference."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    results = []
    for prompt in prompts:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                results.append(resp.choices[0].message.content)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [API ERROR] {e}")
                    results.append("")
                time.sleep(2 ** attempt)
    return results


def run_vllm_inference(prompts: list[str], model_name: str,
                       batch_size: int = 64,
                       max_new_tokens: int = 512) -> list[str]:
    """Fast batched inference using vLLM (recommended for A100/large models).
    Install: pip install vllm
    """
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    print(f"Loading teacher model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Teacher inference (vLLM)"):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            results.append(out.outputs[0].text.strip())
    return results


def run_local_inference(prompts: list[str], model_name: str,
                        batch_size: int = 4,
                        max_new_tokens: int = 512) -> list[str]:
    """Local model inference via HuggingFace transformers (fallback if vLLM unavailable)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",            # auto-splits across all available GPUs
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded on: {[str(p.device) for p in list(model.parameters())[:1]]}")

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Teacher inference"):
        batch = prompts[i:i + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for p in batch]
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for out in outputs:
            generated = out[input_len:]
            results.append(
                tokenizer.decode(generated, skip_special_tokens=True).strip()
            )
    return results
