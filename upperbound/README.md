# Upper Bound Evaluation

Evaluate a large open-source model on the same HotpotQA benchmark using
simple RAG — retrieve top-k passages for the original question and answer in
one shot. No classification, decomposition, or multi-hop agent logic.

## What this measures

A clean two-way comparison:

```
Large model (32B/70B) + simple RAG       ← this script
        vs.
Fine-tuned Qwen2.5-3B + agentic pipeline ← run_eval.py  (EM=0.560, F1=0.675)
```

- If the large model wins by a large margin: model capability is the bottleneck.
  More GRPO training or a larger base model would help most.
- If the large model wins by a small margin: the agentic pipeline is doing real
  work — the structured decomposition is compensating for the weaker model.
- If the 3B agent beats the large RAG-only model: the agentic design is
  extracting more value than raw scale alone.

## Hardware requirements

| Model | GPUs | VRAM | Est. cost / hr |
|---|---|---|---|
| Qwen/Qwen2.5-14B-Instruct | 1× A100 40GB | ~28 GB | ~$1.50 |
| Qwen/Qwen2.5-32B-Instruct | 1× A100 80GB | ~64 GB | ~$2.50 |
| Qwen/Qwen2.5-72B-Instruct | 2× A100 80GB | ~144 GB | ~$5.00 |
| meta-llama/Llama-3.3-70B-Instruct | 2× A100 80GB | ~140 GB | ~$5.00 |

All figures are bfloat16 with vLLM. 500-question eval takes ~5–15 min on A100 with vLLM.

## Setup

```bash
# On the GPU server — copy the project or clone it, then:
pip install vllm

# Activate your existing venv if copying from local:
source /path/to/venv/bin/activate
pip install vllm
```

vLLM will automatically download the model from HuggingFace on first run.
For gated models (Llama), run `huggingface-cli login` first.

## Usage

```bash
cd upperbound/

# Recommended starting point
python eval_upperbound.py --model Qwen/Qwen2.5-32B-Instruct

# 72B with 2 GPUs
python eval_upperbound.py --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel 2

# Quick smoke test (first 50 questions)
python eval_upperbound.py --model Qwen/Qwen2.5-32B-Instruct --n 50

# Reuse existing FAISS index (if already built by run_eval.py)
python eval_upperbound.py --model Qwen/Qwen2.5-32B-Instruct --load-index
```

Results are saved to `upperbound/results/eval_<model>_rag.json`.
