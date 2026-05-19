# HotpotQA Live Multi-Hop Agent

A live inference multi-hop QA agent built on **LangGraph** and **Qwen2.5-3B-Instruct**,
evaluated on the HotpotQA benchmark. The agent classifies, decomposes, retrieves,
and answers every question entirely at inference time — no pre-computed artifacts.

The GRPO-trained 3B agent achieves **EM=0.504** on 500 HotpotQA questions,
closing most of the gap to a 32B model doing simple RAG (EM=0.554), while
achieving higher context recall (0.794 vs 0.748).

> Full experiment results, design decisions, and future plans: [Analysis.md](Analysis.md)

---

## What is HotpotQA?

HotpotQA requires reasoning over two Wikipedia passages to answer a question.
Two question types:

- **Bridge** — find an intermediate entity, then use it to answer the original question.
  > *"What nationality is the actress who starred in Pretty Woman?"*
  > Step 1: Who starred in Pretty Woman? → Julia Roberts
  > Step 2: What nationality is Julia Roberts? → American

- **Comparison** — compare two entities on a shared property.
  > *"Were Scott Derrickson and Ed Wood of the same nationality?"*
  > Both American → yes

---

## Architecture

### Dual-role model

A single `PeftModel` serves two roles via a context manager:

```python
# Reasoning (classify, decompose, rewrite) — base model, no adapter bias
with model.disable_adapter():
    output = model.generate(...)

# Answer synthesis — GRPO adapter active
output = model.generate(...)
```

The base model is better at open-ended reasoning. The GRPO adapter is trained
to produce short, precise answer phrases. Using one model halves memory usage.

### LangGraph pipeline

**Bridge path:**
```
classify → decompose → retrieve_hop1 → answer_hop1
         → formulate_hop2 → retrieve_hop2 → answer_final → verify
```

**Comparison path:**
```
classify → rewrite → retrieve_comparison → answer_final → verify
```

**Shared retry loop (both paths):**
```
verify fails → local retry (retry_count=0)
             → web search  (retry_count=1)
             → end         (retry_count≥2)
```

If named entities from the question are absent from the corpus
(`uncovered_entities`), the agent skips local retry and goes straight to
Wikipedia web search.

### Retriever

FAISS dense index with `all-MiniLM-L6-v2` embeddings. Each paragraph is
encoded as `"<title>\n<text>"` and indexed with `IndexFlatIP` (cosine similarity).

---

## Project Structure

```
hotpotqa-agent/
├── config.py               Paths, model names, thresholds
├── src/
│   ├── models.py           Load PeftModel + tokenizer
│   ├── reasoner.py         All LLM reasoning calls (classify, decompose, rewrite)
│   ├── retriever.py        FAISS dense retriever with save/load
│   ├── graph.py            LangGraph StateGraph pipeline
│   └── evaluator.py        EM, F1, context recall, faithfulness
├── data/
│   ├── grpo_val.jsonl      500-question HotpotQA evaluation set
│   ├── corpus.jsonl        Paragraph corpus (title + text)
│   └── faiss.index         Saved FAISS index
├── upperbound/
│   ├── eval_upperbound.py  Evaluate large models (14B/32B) as RAG-only baseline
│   └── README.md           Hardware guide for GPU eval
├── run_agent.py            CLI demo: single question
├── run_eval.py             Batch evaluation over grpo_val.jsonl
├── debug_agent.py          Step-by-step trace of a single question
└── analyze_results.py      Failure analysis on eval results
```

---

## Quickstart

### Setup

```bash
pip install -r requirements.txt
```

Set your adapter path in `config.py`:
```python
GRPO_ADAPTER_REPO = "your-hf-username/your-adapter-repo"  # or local path
```

### Run a single question

```bash
python run_agent.py --question "Were Scott Derrickson and Ed Wood of the same nationality?"
```

### Run full evaluation (500 questions)

```bash
python run_eval.py            # builds FAISS index and runs eval
python run_eval.py --n 50     # quick smoke test (first 50 questions)
python run_eval.py --load-index  # reuse saved index
```

Results saved to `results/eval_live.json`.

### Debug a single question (step-by-step trace)

```bash
python debug_agent.py --question "What is the nationality of the director of Schindler's List?"
```

### Upper/lower bound evaluation (GPU server)

```bash
# Lower bound — base 3B + simple RAG
python upperbound/eval_upperbound.py --model Qwen/Qwen2.5-3B-Instruct

# Upper bound — large model + simple RAG (requires A100)
python upperbound/eval_upperbound.py --model Qwen/Qwen2.5-32B-Instruct

# With GRPO adapter loaded (isolates training effect)
python upperbound/eval_upperbound.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter your-hf-username/your-adapter-repo
```

---

## Results Summary

| System | EM | F1 | ctx_recall |
|---|---|---|---|
| 3B Base + RAG (lower bound) | 0.458 | 0.541 | 0.748 |
| 3B GRPO + RAG only | 0.468 | 0.552 | 0.748 |
| **3B GRPO + Agent (ours)** | **0.504** | **0.584** | **0.794** |
| 14B Base + RAG | 0.544 | 0.649 | 0.748 |
| 32B Base + RAG | 0.554 | 0.656 | 0.748 |

The agent's multi-hop decomposition achieves the best context recall across all
model sizes. GRPO training adds +1pp EM; the agent design adds +3.6pp.

See [Analysis.md](Analysis.md) for full run-by-run results, ablation studies,
design decisions, and the future improvement plan.
