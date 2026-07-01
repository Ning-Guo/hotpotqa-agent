# HotpotQA Multi-Hop Agent

A multi-hop question answering agent built with LangGraph + Qwen2.5-3B-Instruct, trained with GRPO and distilled with a Query LoRA adapter, evaluated on HotpotQA distractor split.

All reasoning happens at inference time — no pre-computed chain-of-thought.

---

## Results

Four controlled experiments on the same 500-question eval set (`data/grpo_val.jsonl`):

| Experiment | Model | Retrieval | EM | F1 | ctx_recall |
|---|---|---|---|---|---|
| Exp1: Upper Bound | 32B + Golden passages | None (oracle) | 0.788 | 0.884 | 1.000 |
| Exp2: 32B Agent | 32B + Agent | FAISS TOP_K=5 | 0.640 | 0.709 | 0.955 |
| Exp3: 3B GRPO Agent | 3B + GRPO + Agent | FAISS TOP_K=5 | 0.540 | 0.625 | 0.902 |
| **Exp4: + Query LoRA** | **3B + GRPO + Query LoRA + Agent** | **FAISS TOP_K=5** | **0.550** | **0.639** | **0.931** |

**Key findings:**
- A 3B GRPO-tuned model with multi-hop agent design reaches within 9pp EM of a 32B model
- Query LoRA distillation (32B teacher → 3B student) improves context recall by +2.9pp and sub-query entity grounding by +14pp
- The main gap between 3B and 32B is sub-query decomposition quality, not model capacity for answering

See [`eval/EVAL_ANALYSIS.md`](eval/EVAL_ANALYSIS.md) for full experiment breakdown and failure analysis.

---

## Architecture

**Dual-role model:** A single `PeftModel` (Qwen2.5-3B-Instruct) with two LoRA adapters:
- `set_adapter("grpo")` → fine-tuned for final answer synthesis
- `set_adapter("query")` → fine-tuned for sub-query generation (Query LoRA)
- `disable_adapter()` → base model for classify / answer_hop1

**LangGraph pipeline:**
```
Bridge:     classify → decompose → retrieve_hop1 → answer_hop1
                     → formulate_hop2 → retrieve_hop2 → answer_final → verify

Comparison: classify → rewrite → retrieve_comparison → answer_final → verify

Retry:      verify fails → fallback retrieval → web search → end
```

**Retriever:** FAISS + `BAAI/bge-base-en-v1.5`, TOP_K=5, cosine similarity.

![Agent Graph](results/graph.png)

---

## Project Structure

```
├── src/
│   ├── graph.py            LangGraph pipeline (bridge + comparison + verify/retry)
│   ├── reasoner.py         LLM calls: classify, decompose, rewrite, answer
│   ├── retriever.py        FAISS dense retriever (BGE embeddings)
│   ├── evaluator.py        EM, F1, context recall, faithfulness metrics
│   └── models.py           Load PeftModel with GRPO + Query LoRA adapters
├── training/
│   ├── 01_prepare_dataset.py     HotpotQA → SFT/GRPO splits
│   ├── 02_generate_sft_data.py   32B teacher traces for SFT
│   ├── 03_train_sft.py           SFT warm-up
│   ├── 04_merge_adapter.py       Merge SFT adapter into base
│   ├── 05_train_grpo.py          GRPO fine-tuning (reward = 0.5×EM + 0.5×F1)
│   └── query_lora/               Query LoRA distillation pipeline
│       ├── 01_generate_data.py   32B teacher annotates sub_q1/sub_q2 (22K samples)
│       ├── 02_train_query_lora.py  SFT with completion-only loss
│       ├── 03_eval_subq_quality.py Sub-query quality diagnostic
│       └── 04_eval_e2e.py        End-to-end eval with dual-adapter inference
├── eval/
│   ├── exp1_upperbound.py        32B + golden passages (vLLM batch)
│   ├── exp2_32b_agent.py         32B + RAG + full agent
│   ├── exp3_3b_grpo_agent.py     3B GRPO + RAG + agent
│   └── EVAL_ANALYSIS.md          Full experiment analysis + failure taxonomy
├── results/                      JSON result files for all experiments
├── config.py                     Paths, model names, thresholds
├── run_agent.py                  Single question CLI
├── run_eval.py                   Batch evaluation
├── serve.py                      Gradio demo UI
└── debug_agent.py                Step-by-step agent trace
```

---

## Setup

```bash
pip install -r requirements.txt
```

Models are loaded from HuggingFace automatically:
- Base: `Qwen/Qwen2.5-3B-Instruct`
- GRPO adapter: `Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter`
- Query LoRA: `Norm11/qwen2.5-3b-query-lora` *(uploaded separately)*
- Embedding: `BAAI/bge-base-en-v1.5`

---

## Run

```bash
# Single question
python run_agent.py --question "Were Scott Derrickson and Ed Wood of the same nationality?"

# Batch eval (500 questions, reuse saved index)
python run_eval.py --load-index --output results/eval.json

# Step-by-step debug
python debug_agent.py --question "..."
```

**Local demo UI** (Gradio):
```bash
python serve.py --load-index
# Open http://localhost:7860
```
Shows each agent step in real time: classify → decompose → retrieve → answer → verify, with keyword highlighting on retrieved passages.

---

## Training (GPU required, A100 80GB)

**GRPO training pipeline:**
```bash
python training/01_prepare_dataset.py
python training/02_generate_sft_data.py
python training/03_train_sft.py
python training/04_merge_adapter.py
torchrun --nproc_per_node=4 training/05_train_grpo.py --epochs 2 --max-samples 30000
```

**Query LoRA distillation:**
```bash
# Generate 22K training samples (32B teacher via vLLM)
python training/query_lora/01_generate_data.py

# Train LoRA adapter (~3 hours, 1× A100)
python training/query_lora/02_train_query_lora.py

# Evaluate sub-query quality improvement
python training/query_lora/03_eval_subq_quality.py \
    --adapter training/query_lora/checkpoints/query_lora/final --load-index

# End-to-end eval vs Exp3 baseline
python training/query_lora/04_eval_e2e.py \
    --query-adapter training/query_lora/checkpoints/query_lora/final
```

Trained adapters and datasets are available on HuggingFace:
- GRPO adapter: `Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3`
- Training data: `Norm11/qwen2.5-3b-sft-grpo-hotpotqa-dataset`
