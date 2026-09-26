# Code Walkthrough — How to Present This Project

A structured guide for walking through the codebase, from high-level understanding to deep familiarity. Each stage builds on the previous one.

---

## Stage 1 — Get the Big Picture

**Goal:** Understand what the project does and why, before touching any code.

### Files to read (in order):

**`README.md`**
- What problem we're solving: multi-hop QA on HotpotQA distractor split
- Two-phase experiment story (MiniLM → BGE): why switching the embedder changed everything
- Key numbers to internalize: Exp0 EM=0.572, Exp6 EM=0.592, Exp3 EM=0.538, Exp1 (upper bound) EM=0.786

**`eval/EVAL_ANALYSIS.md`**
- Read the Key Findings section only — skip case studies for now
- The core insight: "the multi-hop agent was solving a retrieval problem, not a reasoning problem"
- Once BGE fixed retrieval (ctx_recall 0.748 → 0.907), the agent's error propagation became the bottleneck

**After Stage 1, you should be able to answer:**
- What is HotpotQA? What are the two question types?
- Why did we build a multi-hop agent? Why does it sometimes hurt?
- What do EM, F1, ctx_recall mean?

---

## Stage 2 — Run the System

**Goal:** See the pipeline actually work before reading implementation.

```bash
# Single question, step-by-step trace
python debug_agent.py --question "Were Scott Derrickson and Ed Wood of the same nationality?"

# Or the Gradio demo
python serve.py --load-index
# Open http://localhost:7860
```

Watch the output and identify the steps: classify → decompose → retrieve → answer → verify.
This gives you a concrete mental model before reading the code.

---

## Stage 3 — Configuration and Entry Points

**Goal:** Understand how the system is wired together at the top level.

### `config.py` (root)
The single source of truth for all paths and model names:
- `EMBEDDING_MODEL`, `INDEX_PATH`, `CORPUS_PATH`
- `BASE_MODEL`, `GRPO_ADAPTER`, `QUERY_LORA_ADAPTER`
- `TOP_K`, `VERIFY_THRESHOLD`, `FALLBACK_THRESHOLD`

Read this first — it tells you what models and files everything depends on.

### `run_agent.py`
Minimal CLI wrapper. Shows how `graph.py` is called with a single question and how the final state is printed. ~50 lines, easy to read.

### `run_eval.py`
Batch evaluation wrapper. Shows how 500 questions are fed through the pipeline and results written to JSON.

---

## Stage 4 — The Core Pipeline 

This is the heart of the project. Read these three files together.

### `src/graph.py` — The State Machine

Start here. This file defines the entire agent pipeline as a LangGraph `StateGraph`.

**Key concepts to understand:**

1. **`QAState` (TypedDict)** — the shared state passed between all nodes:
   ```
   question, q_type, sub_q1, sub_q2, hop1_answer,
   retrieved_passages, final_answer, verified, ...
   ```
   Everything a node needs to know is in here. Nodes read from state, write to state.

2. **Node factories** — each node is a function that takes `state` and returns a dict of updates:
   ```python
   def classify_node(state): ...   → {"q_type": "bridge"}
   def decompose_node(state): ...  → {"sub_q1": "...", "sub_q2_template": "..."}
   def retrieve_hop1_node(state): ... → {"retrieved_passages": [...]}
   ```

3. **Conditional edges** — `should_retry` decides whether to go to fallback or end:
   ```python
   graph.add_conditional_edges("verify", should_retry, {
       "fallback": "retrieve_fallback",
       "end": END,
   })
   ```

4. **Verify node** — read this carefully:
   - For normal answers: token overlap between final_answer and retrieved passages (threshold 0.3)
   - For yes/no answers: entity coverage check
   - Failure → fallback retrieval → web search

5. **`uncovered_entities`** — capitalized tokens in sub_q2/rewritten queries not found in retrieved passages. Used to guide fallback retrieval.

**Draw the graph as you read:**
```
Bridge:     classify → decompose → retrieve_hop1 → answer_hop1
                     → formulate_hop2 → retrieve_hop2 → answer_final → verify

Comparison: classify → rewrite → retrieve_comparison → answer_final → verify

Retry:      verify fails → retrieve_fallback → web_search → end
```

### `src/reasoner.py` — All LLM Calls

Every place the model is called. Read after graph.py so you know what each function is supposed to do.

**Key patterns:**

- `classify()` — keyword heuristic first (fast path), then few-shot LLM call if ambiguous
- `decompose_bridge()` — generates sub_q1 and sub_q2 template with `{entity}` placeholder
- `answer_hop1()` — uses base model (`disable_adapter()`), short extractive answer from passages
- `formulate_hop2()` — fills the `{entity}` in sub_q2 template with hop1_answer
- `answer_final()` — uses GRPO adapter (`set_adapter("grpo")`), produces concise final answer
- `rewrite_comparison()` — rewrites original question into two parallel sub-questions

Notice the adapter switching pattern:
```python
model.disable_adapter()    # for classify, decompose, answer_hop1
model.set_adapter("grpo")  # for answer_final
model.set_adapter("query") # for sub-query generation (Query LoRA)
```

### `src/retriever.py` — FAISS Dense Retrieval

**Key concepts:**

- `IndexFlatIP` — exact search (not ANN/HNSW), cosine similarity via L2 normalize + inner product
- BGE query prefix: `"Represent this sentence for searching relevant passages: "` — asymmetric encoding (queries get prefix, documents don't)
- `union_retrieve()` — merges hop1 + hop2 retrieved passages by deduplication, used in answer_final
- ~4,968 paragraphs in corpus, built once and saved as `data/faiss.index`

---

## Stage 5 — Evaluation (20 min)

**Goal:** Understand how results are measured.

### `src/evaluator.py`

Four metrics — understand what each actually measures:

| Metric | What it checks | Input |
|---|---|---|
| `ctx_recall` | Are gold passage **titles** in top-K results? | retrieved titles vs gold_titles |
| `ans_coverage` | Is gold answer text in retrieved **titles + text**? | retrieved title+text vs gold answer |
| `faithfulness` | Is final answer grounded in retrieved **text only**? | retrieved text vs final answer |
| `ctx_precision` | Are retrieved passages gold? | requires `is_gold` field — always null |

Note: `ctx_recall` is title-level only — a passage counts as retrieved if its title appears, regardless of text content.

### `eval/exp6_grpo_naive_rag.py` (recommended starting eval script)

The simplest eval: single-hop RAG + GRPO model. No agent, no decomposition.
Shows the full eval loop: load index → retrieve → answer → compute metrics → save JSON.

### `eval/exp3_3b_grpo_agent.py`

Full agent eval. Compare with exp6 to see exactly what code the agent adds.

---

## Stage 6 — Training Pipeline 

Only needed if you want to understand how the model was trained.

### Read in order:

**`training/config.py`** — all hyperparameters in one place. Read first.

**`training/01_prepare_dataset.py`** — how HotpotQA is split into SFT/GRPO sets, and how gold_absent_fraction works (10% of examples have gold passage removed to train robustness).

**`training/02_generate_sft_data.py`** — 32B teacher generates `<think>...</think><answer>...</answer>` CoT traces for the 3B student. vLLM batch inference.

**`training/03_train_sft.py`** — LoRA SFT.
- `DataCollatorForSeq2Seq` with `labels=-100` for prompt tokens (completion-only loss)
- LoRA rank=16, alpha=32, all projection modules
- lr=2e-4, 3 epochs, effective batch=32

**`training/05_train_grpo.py`** — GRPO fine-tuning.
- `GRPOTrainer` from TRL
- Reward = format_gate × (0.70×accuracy + 0.25×grounding + length_penalty)
- num_generations=2 (wanted 4, cut to 2 for 80GB VRAM)
- KL coef=0.05 (prevents drifting from SFT reference)

**`training/utils/metrics.py`** — the four reward components. Read this alongside 05_train_grpo.py.
- `reward_format`: hard binary gate on `<answer>` tags
- `reward_accuracy`: 0.5×EM + 0.5×F1
- `reward_grounding`: word overlap of model's intermediate answers (parsed from `<think>`) against gold passage text
- `reward_length`: soft penalty for answers > 15 words

**`training/query_lora/`** — Query LoRA distillation (SFT-based, not RL):
- 32B teacher annotates 22K sub_q1/sub_q2 pairs
- Student trained with completion-only loss on sub-query format
- Evaluated end-to-end in `04_eval_e2e.py` using dual-adapter inference

---

## Stage 7 — Case Studies (20 min)

Go back to `eval/EVAL_ANALYSIS.md` and read the case studies now that you understand the code.

For each case study, trace through the code path:
- Which node generated sub_q1?  →  `reasoner.decompose_bridge()`
- Where was hop1_answer stored?  →  `QAState["hop1_answer"]`
- How did uncovered_entities get computed?  →  `graph.py` verify node
- Why did the agent fail here?  →  error propagation through `formulate_hop2()`

This connects the abstract metrics to concrete code behavior.

---

## Stage 8 — Deep Dives 

For specific questions that come up during a presentation:

| Question | Where to look |
|---|---|
| "How does the FAISS index get built?" | `src/retriever.py` → `build_index()` |
| "What's in the corpus?" | `data/corpus.jsonl` — one paragraph per line, title + text |
| "How is the graph visualized?" | `results/graph.png` — generated by LangGraph's `draw_mermaid_png()` |
| "How does web search fallback work?" | `src/graph.py` → `web_search_node`, Wikipedia MediaWiki API |
| "Why BGE prefix but not for documents?" | Asymmetric encoding — BGE was trained this way for retrieval tasks |
| "Why SFT before GRPO?" | GRPO needs a working format gate; without SFT the model can't produce `<answer>` tags reliably |
| "What's the difference between ctx_recall and ans_coverage?" | ctx_recall = gold title in retrieved set; ans_coverage = gold answer words in retrieved title+text |

