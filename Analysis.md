# HotpotQA Agent — Analysis

---

## Part 1: Iteration History

Each entry covers: motivation → design choices → results → analysis.

---

### v1 — Initial Live Agent (MiniLM)

**Motivation:** Port from offline decomposition (HotpotQA-Eval) to fully live inference.

**Design choices:**
- LangGraph: conditional edges express retry/fallback without nested if-else; state graph is inspectable
- `all-MiniLM-L6-v2`: available, reasonable general-purpose baseline
- Single PeftModel with `disable_adapter()`: halves memory vs two separate models; O(1) switching
- Token-overlap faithfulness verifier (threshold=0.3): lightweight, no extra model

**Results (n=500):**

| Type | EM | F1 | ctx_recall |
|---|---|---|---|
| bridge | 0.482 | 0.570 | — |
| comparison | 0.604 | 0.649 | — |
| **overall** | **0.504** | **0.584** | **0.794** |

vs baselines (all MiniLM): 3B Base RAG EM=0.458, 32B Base RAG EM=0.554. Agent closes 83% of the 3B→32B gap.

**Failure analysis (manual, 20 bridge errors):**
- Circular/malformed sub_q1: ~5 cases — decompose prompt had only 3 examples, all straightforward
- Bridge misclassified as comparison: 8 cases — heuristic triggered on standalone `"both"` in bridge questions → EM=0.000 all 8
- Model returns hop-1 entity as final answer: ~2 cases
- Retrieval miss: remainder

---

### v2 — Prompt Fixes

**Motivation:** Error analysis from v1 identified 3 systematic failures in prompts.

**Changes:**
1. Decompose prompt: 3 → 7 examples; added instruction *"do not restate the original question"*
2. Classification heuristic: removed standalone `"both"` from comparison signals; kept only `"were both"`, `"are both"`, `"did both"`
3. Answer prompt: added instruction to return the final entity, not the intermediate hop-1 entity

**Results:**

| | v1 (n=500) | smoke test (n=50) | v2 full (n=500) |
|---|---|---|---|
| EM | 0.504 | 0.560 | 0.504 |

Smoke test showed +5.6pp but full eval showed no change. Root cause: first 50 questions are systematically easier — the smoke test inflated the result. Prompt fixes are correct and were kept; the real bottleneck was retrieval quality, not prompts.

---

### v3 — BGE Embedding Upgrade

**Motivation:** `all-MiniLM-L6-v2` (22M) was trained for symmetric sentence similarity, not asymmetric QA retrieval. Short factual queries scored poorly against long Wikipedia passages.

**Change:** `all-MiniLM-L6-v2` → `BAAI/bge-base-en-v1.5` (109M, contrastively trained on QA hard negatives). Requires query-time prefix: `"Represent this sentence for searching relevant passages: "`. Dense only — BM25 hybrid tested and reverted (see Part 2, E5).

**Results (n=500):**

| Metric | MiniLM | BGE | Delta |
|---|---|---|---|
| EM | 0.504 | **0.574** | +7.0pp |
| F1 | 0.584 | **0.660** | +7.6pp |
| ctx_recall | 0.794 | **0.887** | +9.3pp |
| bridge EM | 0.482 | **0.575** | +9.3pp |
| comparison EM | **0.604** | 0.571 | -3.3pp |
| hard EM | 0.348 | **0.416** | +6.7pp |

ctx_recall jump (+9.3pp) is the primary driver — retrieval was the bottleneck, not the model. Comparison dropped slightly: `union_retrieve` merges results across multiple rewritten sub-queries; higher per-query precision introduced more noise when fused.

**This is the current best result.** Adapter: `Norm11/qwen2.5-3b-grpo-hotpotqa`.

---

### v4 — SFT + GRPO Retrain

**Motivation:** Original GRPO adapter was trained on ~1,500 RAG-retrieved examples with no SFT warmup. Hypothesis: proper SFT warmup on 20K teacher traces + GRPO on 30K examples would improve performance.

**Pipeline:**
1. SFT on 20K teacher-generated reasoning traces (Qwen2.5-32B via API)
2. Merge SFT adapter → `training/checkpoints/sft_merged` (3.09B params)
3. GRPO on 30K HotpotQA distractor examples; 10% gold-absent (padded to 10 passages, answer="insufficient context")
4. Key config corrections: `GRPO_MAX_PROMPT_LEN` 1536→2400, `GRPO_MAX_NEW_TOKENS` 512→350, `GRPO_NUM_GENERATIONS` 4→2

Training: 4-GPU DDP, A100 80GB, ~44hrs, 936 steps. Final reward≈0.77, KL≈0.04, stable throughout.
Adapter: `Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter`.

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
| Train/eval distribution mismatch | GRPO trained on 10 curated gold+distractor passages; eval uses 5 FAISS-retrieved passages. Model learned to answer from idealised context, not retrieval output. |
| Verbosity regression | Reward used `0.5*EM + 0.5*F1`. F1 rewarded fluent long answers. v4 wraps correct entities in full sentences (e.g. `"Ali ibn... produced one of the first encyclopedias."` instead of `"Ali ibn..."`). Accounts for ~70% of the 35 cases where v3 wins. |
| Gold-absent noise | 10% of training used answer="insufficient context". At eval, ctx_recall=0.90 — this hedge behaviour is never needed, but v4 learned to produce disclaimer phrases under uncertainty. |
| KL anchoring | `KL_coef=0.05` kept v4 close to SFT distribution. Since SFT was misaligned, high KL prevented GRPO from correcting it. |

**What v4 did improve:** constraint-preserving sub_q2 (better multi-candidate filtering), lower retry rate (8 vs 15/500).

**What to fix next:**
- Train GRPO on agent-retrieved passages (not gold distractor set) — closes the train/eval gap
- Use pure EM reward (no F1 component) to suppress verbosity
- Remove gold-absent examples or replace with real retrieval-failure cases
- Increase `num_generations` to 4+ for better advantage estimates

---

## Part 2: Engineering Issues

---

### E1 — LangGraph State Not Cleared → Infinite Retry Loop

**Symptom:** 5 questions hit LangGraph's recursion limit (25 calls) silently, consuming 25× expected model calls.

**Root cause:** `web_search` node processed `uncovered_entities` but didn't return `"uncovered_entities": []` in its output dict. LangGraph merges state by dict key — keys not returned retain their previous value. `should_retry` saw non-empty `uncovered_entities` every iteration.

**Fix:** `web_search` explicitly returns `"uncovered_entities": []`. Added hard cap: `retry_count >= 2 → "end"`.

---

### E2 — yes/no Verifier Always Fails → All Comparison Questions Retry

**Symptom:** ~90% of comparison questions triggered web search on first attempt, replacing correct corpus passages with irrelevant Wikipedia content.

**Root cause:** Faithfulness check: `score = |pred_tokens ∩ context_tokens| / |pred_tokens|`. For `pred="yes"`, `pred_tokens=["yes"]` — "yes" never appears literally in Wikipedia passages → score=0.0 < 0.3 → always retries.

**Fix:** Special-case yes/no: skip token overlap, instead check that retrieved passages mention ≥50% of the named entities from the question.

---

### E3 — GRPO KL Explosion from Prompt Truncation

**Symptom:** First full GRPO run: reward fell 0.74→0.08 by step 300; KL diverged 0.01→3.0; model outputs degenerated to empty `<answer></answer>` tags.

**Root cause:**
```
GRPO_MAX_PROMPT_LEN=1536, but prompt p50=1,459 tokens
→ >50% of prompts truncated mid-passage
→ format gate fires on most rollouts (reward=0)
→ all rollouts equally bad → advantage ≈ 0
→ KL penalty accumulates with no reward signal to balance it
→ model drifts randomly from reference → KL explodes → degenerate outputs
```

Discovered by printing prompt length distribution: `p50=1459, p99=2463` — half the dataset was being silently truncated.

**Fix:** `GRPO_MAX_PROMPT_LEN` 1536→2400 (covers p99); `GRPO_KL_COEF` 0.01→0.05 (stronger constraint during early instability); `GRPO_NUM_GENERATIONS` 4→2 (to fit 2400-token prompts in 80GB VRAM). Second run was stable throughout (reward≈0.77, KL≈0.04).

---

### E4 — Hop-1 Entity Returned as Final Answer

**Symptom:** Bridge questions returning the intermediate entity instead of the final answer. Example: *"Who was the personal secretary of the politician born on Sep 1, 1931?"* → `Cecil Parkinson` (hop-1) instead of `Sara Keays` (correct).

**Root cause:** Final answer prompt contained both hop-1 and hop-2 retrieved passages. Model anchored on the more salient entity (the politician) rather than tracing through to the answer role.

**Fix:** Added to bridge final answer prompt: *"The passages may contain intermediate context — answer the original question directly, not any intermediate entity."*

---

### E5 — BM25 Hybrid Retrieval Hurt Comparison (Reverted)

**Symptom:** Adding BM25+FAISS RRF fusion: bridge EM +2.7pp, comparison EM -7.7pp, ctx_recall -4pp — net negative.

**Root cause:** Comparison sub-queries after rewriting contain common words ("nationality", "same", "profession"). BM25 keyword-matches these high-frequency terms, surfacing irrelevant passages. Dense retrieval handles semantic meaning better for abstract queries.

**Decision:** Reverted to dense-only. Optimal: dense-only for comparison, hybrid for bridge — too complex to implement cleanly in the unified retrieval path.

---

### E6 — NLI Verifier Crashed EM by 18pp (Reverted)

**Symptom:** Replacing token-overlap with `cross-encoder/nli-deberta-v3-small` (threshold=0.5): EM 0.560→0.380; 36/50 correct answers marked unverified; 37/50 triggered web search.

**Root cause:** NLI models expect full-sentence hypotheses. Short factoid answers (`"American"`, `"1943"`) never reach P(entailment)>0.5 against a paragraph regardless of correctness.

**Decision:** Reverted. Fix would require expanding hypothesis to a full sentence before NLI scoring — not pursued.

---

### E7 — Wikipedia API 403 Errors

**Symptom:** All web search calls returned 403 Forbidden.

**Root cause:** MediaWiki API blocks requests with no or generic User-Agent header.

**Fix:** Added `User-Agent: hotpotqa-research/1.0 (academic research)` to all requests. Added `verify=False` for LibreSSL certificate issues on macOS.

---

### E8 — vLLM / TRL Environment Incompatibility

**Symptom:** TRL imports vLLM at startup even when `--use-vllm` is not passed; incompatible version caused import failure blocking all training.

**Root cause:** Training environment: torch 2.4.1 + CUDA 12.4. vLLM required torch≥2.6; TRL 1.x (which supports torch≥2.6) introduced `FSDPModule` incompatible with CUDA 12.4 driver.

**Fix:** Install vLLM only for teacher inference (step 02), then `pip uninstall vllm -y` before SFT/GRPO. Pinned: `transformers==4.47.0, trl==0.15.2, peft==0.13.2, accelerate>=1.0.0`.
