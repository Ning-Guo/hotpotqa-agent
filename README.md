# HotpotQA Multi-Hop QA Agent

Multi-hop QA system on HotpotQA using LangGraph + Qwen2.5-3B-Instruct. All reasoning happens at inference time — no pre-computed artifacts.

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
├── eval/                   Controlled ablation experiments (Exp0–Exp6)
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
pip install -r requirements.txt
```

Adapters on HuggingFace:
- GRPO: `Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter`
- Query LoRA: `Norm11/qwen2.5-3b-querylora-hotpotqa`

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

# Step-by-step debug
python debug_agent.py --question "..."
```

## Local Demo UI

Gradio web app with real-time reasoning steps, retrieved passages, and keyword highlighting.

```bash
python serve.py                # first run — builds FAISS index
python serve.py --load-index   # subsequent runs — reuse saved index
python serve.py --port 8080    # custom port (default: 7860)
```

Open `http://localhost:7860`. The UI shows each agent step as it runs (classify → decompose → retrieve → answer → verify), highlights query keywords in retrieved passages, and labels whether context came from the local vector store or Wikipedia web search.

<!-- screenshot -->

## Agent Graph

Visualise the LangGraph pipeline without loading model weights:

```bash
python show_graph.py
```

Outputs:
- ASCII diagram in terminal
- `results/graph.md` — Mermaid source (paste into https://mermaid.live for an interactive render)
- `results/graph.png` — PNG export (requires network)

<!-- graph image -->

## Training (GPU, A100 80GB)

```bash
source ~/venv_train/bin/activate

# SFT + GRPO
python training/01_prepare_dataset.py
python training/02_generate_sft_data.py
python training/03_train_sft.py
python training/04_merge_adapter.py
torchrun --nproc_per_node=4 training/05_train_grpo.py --epochs 2 --max-samples 30000

# Query LoRA (sub-question distillation from 32B teacher)
python training/query_lora/01_generate_data.py --load-index
python training/query_lora/02_train_query_lora.py
python training/query_lora/03_eval_subq_quality.py --load-index
python training/query_lora/04_eval_e2e.py --load-index
```

---

## Results

The project went through two distinct phases, each revealing a key insight about what actually drives performance.

### Phase 1 — MiniLM Embedding (`all-MiniLM-L6-v2`)

Context recall ≈ 0.748. With a weaker retriever, multi-hop decomposition provides substantial gains.

| System | EM | F1 | ctx_recall |
|---|---|---|---|
| 3B Base + RAG | 0.458 | 0.541 | 0.748 |
| 3B GRPO RAG-only | 0.468 | 0.552 | 0.748 |
| **3B GRPO + Agent** | **0.574** | **0.660** | **0.887** |
| 14B Base + RAG | 0.544 | 0.649 | 0.748 |
| 32B Base + RAG | 0.554 | 0.656 | 0.748 |

**Finding:** Multi-hop agent (+10.6pp over naive RAG) and GRPO training (+1.0pp) together let a 3B model outperform 32B naive RAG by 2.0pp. The agent's decomposition was genuinely useful because retrieval was the bottleneck.

---

### Phase 2 — BGE Embedding (`BAAI/bge-base-en-v1.5`)

Context recall ≈ 0.907 (+15.9pp). Stronger retrieval changes the picture entirely.

| Experiment | EM | F1 | ctx_recall |
|---|---|---|---|
| Exp0: 3B base + Naive RAG | 0.568 | 0.657 | 0.907 |
| Exp3: 3B GRPO + Agent | 0.540 | 0.625 | 0.902 |
| Exp4: 3B GRPO + Query LoRA + Agent | 0.550 | 0.639 | 0.931 |
| Exp5: Adaptive routing (comparison→RAG, bridge→agent) | 0.544 | 0.640 | 0.901 |
| **Exp6: 3B GRPO + Naive RAG** | **0.586** | **0.665** | **0.907** |

**Finding:** BGE retrieval alone lifted the naive RAG baseline to 0.568, already above the multi-hop agent (0.540). The optimal combination is GRPO answer synthesis + single-hop retrieval — no decomposition needed.

**Contribution breakdown (BGE era):**
- GRPO adapter: **+1.8pp** (Exp0 → Exp6)
- Multi-hop pipeline: **−4.6pp** (Exp6 → Exp3)

**Core insight:** The multi-hop agent was solving a retrieval problem. Once BGE made direct retrieval reliable (ctx_recall 0.748 → 0.907), the pipeline's error propagation became the dominant source of failures rather than missing context.

See [eval/EVAL_ANALYSIS.md](eval/EVAL_ANALYSIS.md) for full experiment details and failure analysis.
