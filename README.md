# HotpotQA Live Multi-Hop Agent

Live multi-hop QA agent using LangGraph + Qwen2.5-3B-Instruct, evaluated on HotpotQA. All reasoning happens at inference time — no pre-computed artifacts.

## Dataset

HotpotQA distractor split. Two question types:
- **Bridge** — find an intermediate entity, then answer the original question.
- **Comparison** — compare two entities on a shared property.

Eval set: `data/grpo_val.jsonl` — 500 questions (409 bridge / 91 comparison).

---

## Architecture

**Dual-role model:** Single `PeftModel` (Qwen2.5-3B-Instruct + GRPO LoRA adapter)
- `model.disable_adapter()` → base model for reasoning (classify, decompose, rewrite)
- adapter active → GRPO model for final answer synthesis

**LangGraph pipeline:**
```
Bridge:     classify → decompose → retrieve_hop1 → answer_hop1
                     → formulate_hop2 → retrieve_hop2 → answer_final → verify

Comparison: classify → rewrite → retrieve_comparison → answer_final → verify

Retry:      verify fails → fallback retrieval → web search → end (capped at retry_count=2)
```

**Retriever:** FAISS + `BAAI/bge-base-en-v1.5`, TOP_K=5, cosine similarity.

---

## Project Structure

```
├── config.py               Paths, model names, thresholds
├── src/
│   ├── models.py           Load PeftModel + tokenizer
│   ├── reasoner.py         LLM reasoning calls (classify, decompose, rewrite)
│   ├── retriever.py        FAISS dense retriever
│   ├── graph.py            LangGraph pipeline
│   └── evaluator.py        EM, F1, context recall, faithfulness
├── training/               SFT + GRPO training scripts (01–05)
├── data/
│   ├── grpo_val.jsonl      500-question eval set
│   ├── corpus.jsonl        Paragraph corpus
│   └── faiss.index         FAISS index
├── run_agent.py            Single question CLI
├── run_eval.py             Batch evaluation
├── serve.py                Gradio demo UI
└── debug_agent.py          Step-by-step trace
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set adapter in config.py
GRPO_ADAPTER_REPO = "Norm11/qwen2.5-3b-grpo-hotpotqa"
```

## Run

```bash
# Single question
python run_agent.py --question "Were Scott Derrickson and Ed Wood of the same nationality?"

# Full eval (builds FAISS index)
python run_eval.py --output results/eval.json

# Full eval (reuse saved index)
python run_eval.py --load-index --output results/eval.json

# Quick smoke test (20 questions)
python run_eval.py --n 20 --load-index

# Gradio UI
python serve.py --load-index

# Step-by-step debug
python debug_agent.py --question "..."
```

## Training (GPU, A100 80GB)

```bash
source ~/venv_train/bin/activate
python training/01_prepare_dataset.py
python training/02_generate_sft_data.py
python training/03_train_sft.py
python training/04_merge_adapter.py
torchrun --nproc_per_node=4 training/05_train_grpo.py --epochs 2 --max-samples 30000
```

---

## Best Result

**Adapter:** `Norm11/qwen2.5-3b-grpo-hotpotqa` | **Embedding:** `BAAI/bge-base-en-v1.5`

| System | EM | F1 | ctx_recall |
|---|---|---|---|
| 3B Base + RAG (lower bound) | 0.458 | 0.541 | 0.748 |
| 3B GRPO RAG-only | 0.468 | 0.552 | 0.748 |
| **3B GRPO + Agent (ours)** | **0.574** | **0.660** | **0.887** |
| 14B Base + RAG | 0.544 | 0.649 | 0.748 |
| 32B Base + RAG | 0.554 | 0.656 | 0.748 |

| Type | EM | F1 | n |
|---|---|---|---|
| bridge | 0.570 | 0.657 | 409 |
| comparison | 0.593 | 0.643 | 91 |

A 3B GRPO-tuned model with multi-hop decomposition outperforms 32B simple RAG. Agent design contributes +3.6pp EM; GRPO training adds +1.0pp.

See [Analysis.md](Analysis.md) for full experiment history and engineering notes.
