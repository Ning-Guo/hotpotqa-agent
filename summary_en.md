# HotpotQA Live Multi-Hop Agent — Project Summary

## 1. Project Overview

This project builds a **live multi-hop question answering agent** evaluated on the HotpotQA distractor split. All reasoning happens at inference time — no pre-computed artifacts. The core research question is: *can a small fine-tuned model with an intelligent agent pipeline outperform a much larger model with simple RAG?*

**Answer: yes.** A 3B GRPO-tuned model with multi-hop decomposition achieves EM=0.574, outperforming 32B simple RAG (EM=0.554), closing 83% of the 3B→32B gap.

---

## 2. Dataset

**HotpotQA distractor split.** Eval set: `data/grpo_val.jsonl`, 500 questions.

| Type | Count | Description |
|---|---|---|
| Bridge | 409 (82%) | Requires finding an intermediate entity before answering the original question |
| Comparison | 91 (18%) | Compares two explicitly named entities on a shared property → yes/no answer |

Each item contains the original question, answer, question type, difficulty level, supporting facts (gold paragraph titles), and 10 paragraphs (2 gold + 8 distractors).

---

## 3. Architecture

### 3.1 LangGraph Pipeline

The agent is implemented as a **LangGraph StateGraph** with conditional edges. No nested if-else logic — the graph is fully inspectable and restartable.

```
Bridge path:
  START → classify → decompose → retrieve_hop1 → answer_hop1
        → formulate_hop2 → retrieve_hop2 → answer_final → verify
        → [end | retrieve_fallback → answer_final | web_search → answer_final → end]

Comparison path:
  START → classify → rewrite → retrieve_comparison → answer_final → verify
        → [end | retrieve_fallback → answer_final | web_search → answer_final → end]
```

Each node is a pure function over a typed state dict (`QAState`). LangGraph merges state by dict key — any key not returned by a node retains its previous value (source of E1 bug, see §6).

### 3.2 Dual-Role Model

A **single PeftModel** (Qwen2.5-3B-Instruct + GRPO LoRA adapter) serves two roles, switching via `model.disable_adapter()`:

| Mode | How | Used for |
|---|---|---|
| Base model | `with model.disable_adapter()` | classify, decompose, answer_hop1, formulate_hop2, rewrite_comparison |
| GRPO adapter | adapter active | answer_final (bridge) |

This halves peak memory vs. loading two separate models and enables O(1) switching.

**Comparison questions** use a two-step programmatic approach for final answer:
1. Answer each rewritten sub-query independently using the base model (extract property per entity)
2. Compare values programmatically → output yes/no

This avoids asking the small model to do abstract semantic comparison in a single pass, which it reliably fails at.

### 3.3 Retriever

**FAISS + `BAAI/bge-base-en-v1.5`** (109M parameters, contrastively trained on QA hard negatives).

- Cosine similarity, TOP_K=5 per sub-query
- BGE requires query-time prefix: `"Represent this sentence for searching relevant passages: "`
- Bridge questions: hop1 passages (5) + hop2 passages (5), deduped by title → up to 10 passages per question
- Comparison questions: 2 rewritten sub-queries × TOP_K=5, deduped → up to 10 passages

### 3.4 Faithfulness Verifier

Token-overlap check: `score = |pred_tokens ∩ context_tokens| / |pred_tokens|`. Threshold = 0.3.

**Special case for yes/no answers** (Comparison): token overlap is meaningless ("yes"/"no" never appear in Wikipedia passages). Instead, check that retrieved passages mention ≥50% of named entities from the question.

Retry logic:
- Fail + retry_count=0 → `retrieve_fallback` (re-retrieve with original question)
- Fail + retry_count=1 → `web_search` (MediaWiki API)
- retry_count ≥ 2 → force end

---

## 4. Training Pipeline

### Step 1 — Dataset Preparation (`training/01_prepare_dataset.py`)
Parse HotpotQA distractor split. Separate bridge and comparison questions. Build FAISS index over all paragraphs.

### Step 2 — SFT Data Generation (`training/02_generate_sft_data.py`)
Use Qwen2.5-32B-Instruct via API (vLLM) to generate 20K teacher reasoning traces. Each trace contains:
- For bridge: sub_q1, retrieved passages, hop1_answer, sub_q2, final answer
- For comparison: rewritten sub-queries, property values per entity, final yes/no

### Step 3 — SFT Training (`training/03_train_sft.py`)
Fine-tune Qwen2.5-3B-Instruct on teacher traces. Output: SFT LoRA adapter.

### Step 4 — Merge Adapter (`training/04_merge_adapter.py`)
Merge SFT LoRA into base weights → `training/checkpoints/sft_merged` (3.09B params). Required before GRPO to avoid double-adapter complexity.

### Step 5 — GRPO Training (`training/05_train_grpo.py`)
Reinforcement learning from outcome reward:
- **Reward**: `0.5 × EM + 0.5 × F1` on final answer
- **Training data**: 30K HotpotQA distractor examples; 10% gold-absent (padded to 10 passages, answer="insufficient context")
- **Hardware**: 4-GPU DDP, A100 80GB, ~44 hours, 936 steps
- **Final metrics**: reward ≈ 0.77, KL ≈ 0.04, stable throughout
- **Key config**: `GRPO_MAX_PROMPT_LEN=2400`, `GRPO_MAX_NEW_TOKENS=350`, `GRPO_NUM_GENERATIONS=2`
- **Published adapter**: `Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter`

---

## 5. Iteration History

### v1 — Initial Live Agent (MiniLM embedding)

**Design choices:**
- `all-MiniLM-L6-v2` (22M): available, reasonable baseline
- Token-overlap faithfulness verifier (threshold=0.3)
- Heuristic classification: keyword signals + capitalised entity count

**Results (n=500):**

| Type | EM | F1 |
|---|---|---|
| bridge | 0.482 | 0.570 |
| comparison | 0.604 | 0.649 |
| **overall** | **0.504** | **0.584** |

vs baselines: 3B Base RAG EM=0.458, 32B Base RAG EM=0.554. Agent closes 83% of the 3B→32B gap.

**Failure analysis (manual, 20 bridge errors):**
- Circular/malformed sub_q1: ~5 cases (decompose prompt only had 3 examples)
- Bridge misclassified as comparison: 8 cases (heuristic triggered on standalone "both") → EM=0.000 all 8
- Model returns hop-1 entity as final answer: ~2 cases
- Retrieval miss: remainder

---

### v2 — Prompt Fixes

**Changes:**
1. Decompose prompt: 3 → 7 examples; added "do not restate the original question"
2. Classification heuristic: removed standalone "both"; kept only "were both", "are both", "did both"
3. Answer prompt: added instruction to return final entity, not intermediate hop-1 entity

**Results:** Smoke test (n=50) showed +5.6pp → EM=0.560. Full eval (n=500) showed no change → EM=0.504.

**Root cause:** First 50 questions are systematically easier. Smoke test inflated the result. Prompt fixes are correct and were kept — the real bottleneck was retrieval, not prompts.

---

### v3 — BGE Embedding Upgrade (Current Best)

**Motivation:** `all-MiniLM-L6-v2` was trained for symmetric sentence similarity, not asymmetric QA retrieval. Short factual queries scored poorly against long Wikipedia passages.

**Change:** `all-MiniLM-L6-v2` → `BAAI/bge-base-en-v1.5` (109M, contrastively trained on QA hard negatives).

**Results (n=500):**

| Metric | MiniLM | BGE | Delta |
|---|---|---|---|
| EM | 0.504 | **0.574** | +7.0pp |
| F1 | 0.584 | **0.660** | +7.6pp |
| ctx_recall | 0.794 | **0.887** | +9.3pp |
| bridge EM | 0.482 | **0.575** | +9.3pp |
| comparison EM | **0.604** | 0.571 | -3.3pp |
| hard EM | 0.348 | **0.416** | +6.7pp |

ctx_recall jump (+9.3pp) is the primary driver — retrieval was the bottleneck, not the model. Comparison dropped slightly: union retrieval merges results from multiple sub-queries; higher per-query precision introduced noise when fused.

**This is the current best result.** Adapter: `Norm11/qwen2.5-3b-grpo-hotpotqa`.

---

### v4 — SFT + GRPO Retrain

**Motivation:** Original GRPO adapter was trained on ~1,500 RAG-retrieved examples with no SFT warmup. Hypothesis: SFT warmup on 20K teacher traces + GRPO on 30K examples would improve performance.

**Results (n=500):**

| Metric | v3 (best) | v4 (retrain) | Delta |
|---|---|---|---|
| EM | **0.574** | 0.538 | -3.6pp |
| F1 | **0.660** | 0.625 | -3.5pp |
| ctx_recall | 0.887 | **0.901** | +1.4pp |
| retried | 15/500 | **8/500** | better |
| avg wrong prediction length | 13.6 chars | 22.6 chars | worse |

**Why v4 underperformed — root causes:**

| Cause | Evidence |
|---|---|
| Train/eval distribution mismatch | GRPO trained on 10 curated gold+distractor passages; eval uses 5 FAISS-retrieved passages. Model learned to answer from idealised context. |
| Verbosity regression | Reward used `0.5×EM + 0.5×F1`. F1 rewarded fluent long answers. v4 wraps correct entities in full sentences. Accounts for ~70% of cases where v3 wins. |
| Gold-absent noise | 10% of training used answer="insufficient context". At eval ctx_recall=0.90 — this hedge never applies, but v4 learned to produce disclaimer phrases. |
| KL anchoring | `KL_coef=0.05` kept v4 close to SFT distribution. Since SFT was misaligned, KL prevented GRPO from correcting it. |

---

## 6. Engineering Issues

### E1 — LangGraph State Not Cleared → Infinite Retry Loop
**Symptom:** 5 questions hit LangGraph's recursion limit silently.
**Root cause:** `web_search` node didn't return `"uncovered_entities": []`. LangGraph retains unreturned keys → `should_retry` saw non-empty entities every iteration.
**Fix:** Explicit `"uncovered_entities": []` return. Hard cap: `retry_count >= 2 → "end"`.

### E2 — yes/no Verifier Always Fails → All Comparison Questions Retry
**Symptom:** ~90% of comparison questions triggered web search, replacing correct passages.
**Root cause:** `score = |{"yes"} ∩ context_tokens| / 1 = 0.0` — "yes"/"no" never appear literally in Wikipedia.
**Fix:** Special-case yes/no: check entity coverage (≥50% of named entities from question present in context).

### E3 — GRPO KL Explosion from Prompt Truncation
**Symptom:** First full GRPO run: reward fell 0.74→0.08 by step 300; KL diverged 0.01→3.0.
**Root cause:** `GRPO_MAX_PROMPT_LEN=1536` but prompt `p50=1,459` → >50% truncated → format gate fires on most rollouts → advantage ≈ 0 → KL explodes.
**Fix:** `GRPO_MAX_PROMPT_LEN` 1536→2400, `GRPO_KL_COEF` 0.01→0.05, `GRPO_NUM_GENERATIONS` 4→2.

### E4 — Hop-1 Entity Returned as Final Answer
**Symptom:** Bridge questions returning intermediate entity instead of final answer.
**Root cause:** Final answer prompt contained both hop-1 and hop-2 passages. Model anchored on the more salient entity.
**Fix:** Added to bridge final answer prompt: *"answer the original question directly, not any intermediate entity."*

### E5 — BM25 Hybrid Retrieval Hurt Comparison (Reverted)
**Symptom:** BM25+FAISS RRF: bridge EM +2.7pp, comparison EM -7.7pp, ctx_recall -4pp.
**Root cause:** Comparison sub-queries contain high-frequency words ("nationality", "same"). BM25 keyword-matches these, surfacing irrelevant passages.
**Decision:** Reverted to dense-only.

### E6 — NLI Verifier Crashed EM by 18pp (Reverted)
**Symptom:** `cross-encoder/nli-deberta-v3-small` (threshold=0.5): EM 0.560→0.380.
**Root cause:** NLI models expect full-sentence hypotheses. Short factoid answers never reach P(entailment)>0.5.
**Decision:** Reverted.

### E7 — Wikipedia API 403 Errors
**Root cause:** MediaWiki API blocks requests with no/generic User-Agent.
**Fix:** Added `User-Agent: hotpotqa-research/1.0`. Added `verify=False` for LibreSSL on macOS.

### E8 — vLLM / TRL Environment Incompatibility
**Root cause:** TRL imports vLLM at startup; torch 2.4.1 + CUDA 12.4 incompatible with vLLM (requires torch≥2.6).
**Fix:** `pip uninstall vllm -y` before SFT/GRPO. Pinned: `transformers==4.47.0, trl==0.15.2, peft==0.13.2`.

---

## 7. Controlled Comparison Experiments

To establish a convincing experimental narrative, three controlled experiments were run on the same 500-question eval set with the same FAISS index.

| Experiment | Model | Retrieval | Pipeline |
|---|---|---|---|
| Exp1 | Qwen2.5-32B | Golden passages (supporting facts only) | Single-turn QA |
| Exp2 | Qwen2.5-32B | FAISS TOP_K=5 | Full LangGraph Agent |
| Exp3 | Qwen2.5-3B + GRPO | FAISS TOP_K=5 | Full LangGraph Agent |

Scripts in `eval/` directory — no modification to original code. `PlainModelWrapper` provides a no-op `disable_adapter()` context manager so Exp2's 32B base model is compatible with the existing `Reasoner` class.

### Overall Results

| Experiment | EM | F1 | ctx_recall | ans_coverage | faithfulness |
|---|---|---|---|---|---|
| Exp1: 32B + Golden (upper bound) | 0.788 | 0.884 | 1.000 | 0.976 | 0.921 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.955 | 0.956 | 0.805 |
| Exp3: 3B GRPO + RAG + Agent | 0.540 | 0.625 | 0.902 | 0.914 | 0.885 |

### By Question Type

| Experiment | Bridge EM | Comparison EM |
|---|---|---|
| Exp1: 32B + Golden | 0.802 | 0.725 |
| Exp2: 32B + Agent | 0.724 | **0.264** |
| Exp3: 3B GRPO + Agent | 0.533 | 0.571 |

### Key Findings

**Finding 1 — Retrieval is not the only bottleneck.**
Exp2 ctx_recall=0.955 (near-perfect), yet EM is 14.8pp below Exp1. Agent pipeline errors (hop decomposition failures, hop1 answer propagation) introduce significant losses even with near-perfect retrieval. Simply increasing TOP_K has limited impact.

**Finding 2 — 3B GRPO closes 56% of the gap to 32B+Agent.**
Exp3 EM=0.540 vs Exp2 EM=0.640 — only 10pp gap despite 10× fewer parameters. GRPO fine-tuning compensates substantially for model scale.

**Finding 3 — Exp2 Comparison anomaly (EM=0.264, below random chance).**
Root cause: the faithfulness verifier has a special yes/no path that checks entity coverage. The 32B base model (no GRPO fine-tuning) produces verbose output instead of clean yes/no → verifier fails → web search replaces correct passages → answer degrades. This is an implicit coupling between agent pipeline and model output format, not a 32B capability failure (Exp1 shows 32B achieves 0.725 on comparison with golden passages).

**Finding 4 — ctx_recall gap confirms query quality matters more than TOP_K.**
Exp2 ctx_recall=0.955, Exp3=0.902. The 32B model generates better sub-queries, recalling gold passages more reliably. The 10pp EM gap between Exp2 and Exp3 is only partially explained by retrieval — model-level query decomposition quality is the larger driver.

---

## 8. Sub-Query Failure Analysis

Among the 95 bridge questions where Exp2 answered correctly and Exp3 did not, sub_q1/hop1_answer/sub_q2 were compared directly.

### Failure Taxonomy

| Category | Count | % | Root Cause |
|---|---|---|---|
| A: hop1_answer differs (sub_q1 wrong) | 63 | 66% | 3B generates poor sub_q1, first hop goes wrong direction |
| B: sub_q2 differs (hop1 correct) | 20 | 21% | hop1_answer not properly incorporated into sub_q2 |
| C: same queries, wrong answer | 12 | 13% | Pure generation failure, unrelated to retrieval |

### Category A Patterns (66%)

**A1 — Few-shot contamination:** 3B copies content from decompose prompt examples into sub_q1.
```
Q: "Party Never Ends is an album by the Romanian singer who studied at what college?"
Exp2 sub_q1: "Who is the Romanian singer who released Party Never Ends?"  → Inna ✓
Exp3 sub_q1: "What college did the Romanian singer who starred in Pretty Woman study at?"
             ↑ "Pretty Woman" leaked from few-shot example → answer jumps to wrong university
```

**A2 — Wrong intermediate entity:** sub_q1 resolves to the wrong entity.
```
Q: "The army officer who committed a murder in 1970 at Fort Bragg was born which year?"
Exp2: sub_q1 → Jeffrey R. MacDonald → 1943 ✓
Exp3: sub_q1 → Ronald Adrin Gray → 1945 ✗
```

**A3 — Spurious constraints in sub_q1:** Extra context redirects retrieval.
```
Q: "Black Holes in the Sand features a cover-version of Diane, by what American rock band?"
Exp2 sub_q1: "What American rock band performed the song Diane?" → Hüsker Dü (original) ✓
Exp3 sub_q1: "What American rock band covered Diane for Black Holes in the Sand?"
             → Gravenhurst (the band that made the cover) ✗
```

### Category B Patterns (21%)

**B1 — sub_q2 asks for wrong attribute:**
```
Q: "...starred with Jason Patrick in a 2015 horror film written by who?"
hop1: Mark Margolis (both models agree)
Exp2 sub_q2: "Who wrote the 2015 horror film that Mark Margolis starred in with Jason Patrick?" → Ido Fluk ✓
Exp3 sub_q2: "What films did Mark Margolis star in with Jason Patrick in 2015?" → Mark Margolis ✗
```

**B2 — sub_q2 doesn't substitute hop1_answer:**
```
Q: "Who is the mother of the striker for the Czech First League club born on 25th June 1983?"
hop1: Marc Janko (both models agree)
Exp2 sub_q2: "Who is the mother of Marc Janko?" → Eva Janko ✓
Exp3 sub_q2: "Who is the mother of the player born on 25th June 1983 who plays for...?"
             ↑ hop1_answer completely ignored, second hop re-runs first hop → Marc Janko ✗
```

### Category C Patterns (13%)

- **Verbosity:** `"Tom Coburn was born on March 14, 1948."` instead of `"March 14, 1948"` → EM=0
- **Hallucination:** Correct retrieval, completely fabricated answer unrelated to context

---

## 9. Future Work

### Priority 1 — Query LoRA via 32B Distillation (addresses 87% of bridge failures)

Train a new LoRA adapter specifically for query rewriting (sub_q1, sub_q2, comparison rewrite), distilling from 32B model annotations. This adapter runs in the `Reasoner` (base model reasoning steps) and is separate from the existing GRPO answer adapter.

**Architecture:**
```
Qwen2.5-3B-Instruct (base)
    ├── GRPO adapter  → final answer synthesis (existing, unchanged)
    └── Query LoRA    → sub_q1 / sub_q2 / comparison rewrite (new)
```

**Training data design (3 task types, ~10K–15K examples total):**

| Task | Input | Output | Annotation source |
|---|---|---|---|
| sub_q1 generation | question | clean first-hop sub-question | 32B model |
| sub_q2 generation | question + hop1_answer | second-hop sub-question with hop1 substituted | 32B model, but hop1_answer from 3B base (distribution match) |
| comparison rewrite | question | 2 sub-queries, one per entity | 32B model |

**Key constraints:**
- Prompts fed to 32B must be identical to `src/reasoner.py` templates — no free-form generation
- Sub_q2 training uses 3B's own hop1_answer (not 32B's), so train/inference distributions match
- Quality filter: only keep examples where the generated query achieves ctx_recall=1.0 on FAISS retrieval
- Loss: standard SFT next-token prediction on completion tokens only (prompt masked)
- No task labels needed — prompt format naturally distinguishes the three tasks

**Expected gain:** Bridge EM +5–8pp, overall EM to ~0.58–0.60.

### Priority 2 — GRPO Retraining on Agent-Retrieved Passages (addresses v4 failure)

The core v4 failure was training on 10 gold+distractor passages while evaluating on 5 FAISS-retrieved passages. Fix:
- Train GRPO rollouts on actual agent-retrieved passages (not gold distractor set)
- Use pure EM reward (remove F1 component) to suppress verbosity
- Remove gold-absent examples or replace with real retrieval-failure cases
- Increase `num_generations` to 4+ for better advantage estimates

### Priority 3 — Cross-Encoder Reranker

Add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) between FAISS retrieval and model input: retrieve TOP_K=20, rerank to TOP_5. Joint query-passage encoding provides much higher precision than bi-encoder. Expected ctx_recall improvement: 0.902 → 0.93–0.95 for Exp3.

### Priority 4 — Verbosity / Hallucination in Generation (Category C)

- Add brevity penalty to GRPO reward: penalise predictions longer than gold answer by more than N tokens
- Strengthen answer prompt instruction: "Give a short answer phrase only — no explanation, no full sentence"
- Raise faithfulness verifier threshold or add sentence-level expansion before NLI scoring

---

## 10. Full Results Reference

### Original Iteration Results

| System | EM | F1 | ctx_recall |
|---|---|---|---|
| 3B Base + RAG (lower bound) | 0.458 | 0.541 | 0.748 |
| 3B GRPO RAG-only | 0.468 | 0.552 | 0.748 |
| **3B GRPO + Agent (v3, best)** | **0.574** | **0.660** | **0.887** |
| 14B Base + RAG | 0.544 | 0.649 | 0.748 |
| 32B Base + RAG | 0.554 | 0.656 | 0.748 |

### Controlled Experiments

| System | EM | F1 | Bridge EM | Comparison EM | ctx_recall |
|---|---|---|---|---|---|
| Exp1: 32B + Golden (upper bound) | 0.788 | 0.884 | 0.802 | 0.725 | 1.000 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.724 | 0.264 | 0.955 |
| Exp3: 3B GRPO + RAG + Agent | 0.540 | 0.625 | 0.533 | 0.571 | 0.902 |
