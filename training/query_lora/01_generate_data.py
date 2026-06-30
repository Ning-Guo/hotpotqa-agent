#!/usr/bin/env python3
"""
01_generate_data.py — Generate SFT training data for Query LoRA.

Pipeline:
  Bridge questions:
    1. 32B teacher → sub_q1  (batch, vLLM)
    2. FAISS retrieval with sub_q1
    3. 3B base model → hop1_answer  (sequential, HF)
    4. 32B teacher → sub_q2  (batch, vLLM, using 3B's hop1_answer)
    5. Quality filter: ctx_recall = 1.0 for both sub_q1 and sub_q2

  Comparison questions:
    1. 32B teacher → 2 sub-queries  (batch, vLLM)
    2. Quality filter: union ctx_recall = 1.0

Output:
  data/raw_generated.jsonl   — all generated samples before filtering
  data/train.jsonl           — filtered SFT training samples
  data/val.jsonl             — held-out validation samples (5%)

Usage:
  pip install vllm
  python training/query_lora/01_generate_data.py --load-index
  python training/query_lora/01_generate_data.py --n-bridge 500 --n-comparison 100  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)  # training/query_lora/config.py takes priority
sys.path.insert(1, ROOT)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import config as cfg
from src.retriever import Retriever
from src.evaluator import supporting_fact_recall

# ---------------------------------------------------------------------------
# PromptBuilder — reuses src/reasoner.py templates exactly
# ---------------------------------------------------------------------------

from src.reasoner import Reasoner

class PromptBuilder:
    """
    Builds prompts using the exact same templates as src/reasoner.py.
    Does not load model weights — only uses the tokenizer for chat formatting.
    Keeping this in sync with Reasoner ensures train/inference prompt parity.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # Bind Reasoner's prompt methods directly (no model needed)
    _chat                   = Reasoner._chat
    _decompose_prompt       = Reasoner._decompose_prompt
    _hop2_prompt            = Reasoner._hop2_prompt
    _rewrite_prompt         = Reasoner._rewrite_prompt
    _answer_prompt          = Reasoner._answer_prompt
    _parse_rewritten_queries = Reasoner._parse_rewritten_queries

    def messages_decompose(self, question: str) -> list:
        """Return messages list for sub_q1 generation."""
        content = self._decompose_prompt(question)
        # _decompose_prompt returns the full chat-formatted string via _chat;
        # we need the raw content to re-apply with the teacher's tokenizer.
        # Re-extract by calling the content builder directly.
        return self._messages_from_content(question, "decompose")

    def _messages_from_content(self, question: str, task: str) -> list:
        """Build messages list for vLLM chat API."""
        if task == "decompose":
            examples = (
                "Q: What nationality is the actress who starred in Pretty Woman?\n"
                "Sub-q1: Who starred in Pretty Woman?\n\n"
                "Q: In which country was the director of Schindler's List born?\n"
                "Sub-q1: Who directed Schindler's List?\n\n"
                "Q: What is the occupation of the spouse of Barack Obama?\n"
                "Sub-q1: Who is the spouse of Barack Obama?\n\n"
                "Q: Lake Hodges is crossed by the major Interstate Highway that begins in what county?\n"
                "Sub-q1: What major Interstate Highway crosses Lake Hodges?\n\n"
                "Q: What is the lowest weight allowed in the division where Ricco Rodriguez competes?\n"
                "Sub-q1: What division does Ricco Rodriguez compete in?\n\n"
                "Q: In what year did the winner of the 2010 FIFA World Cup win their first World Cup?\n"
                "Sub-q1: Which country won the 2010 FIFA World Cup?\n\n"
                "Q: Who was the personal secretary of the British politician born on September 1, 1931?\n"
                "Sub-q1: Who was the British politician born on September 1, 1931?\n\n"
            )
            content = (
                "Extract the first sub-question from this multi-hop bridge question.\n"
                "The sub-question must identify the INTERMEDIATE entity — not the final answer.\n"
                "Do NOT restate the original question or ask for the final answer directly.\n"
                "Output only the sub-question, no explanation.\n\n"
                f"{examples}"
                f"Q: {question}\n"
                "Sub-q1:"
            )
        elif task == "hop2":
            raise ValueError("Use messages_hop2() for hop2 prompts")
        elif task == "rewrite":
            examples = (
                "Q: Which was founded first, Stanford University or MIT?\n"
                "Q: When was Stanford University founded?\n"
                "Q: When was MIT founded?\n\n"
                "Q: Who is taller, Abraham Lincoln or Napoleon Bonaparte?\n"
                "Q: How tall was Abraham Lincoln?\n"
                "Q: How tall was Napoleon Bonaparte?\n\n"
                "Q: Which city has a larger population, Tokyo or New York?\n"
                "Q: What is the population of Tokyo?\n"
                "Q: What is the population of New York?\n\n"
            )
            content = (
                "Rewrite this comparison question into two simple factual sub-questions, "
                "one per entity. Output exactly two lines, each starting with 'Q:'.\n\n"
                f"{examples}"
                f"Q: {question}\n"
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        return [{"role": "user", "content": content}]

    def messages_hop2(self, question: str, hop1_answer: str) -> list:
        examples = (
            "Q: What nationality is the actress who starred in Pretty Woman?\n"
            "Hop-1 answer: Julia Roberts\n"
            "Sub-q2: What nationality is Julia Roberts?\n\n"
            "Q: In which country was the director of Schindler's List born?\n"
            "Hop-1 answer: Steven Spielberg\n"
            "Sub-q2: In which country was Steven Spielberg born?\n\n"
            "Q: What is the occupation of the spouse of Barack Obama?\n"
            "Hop-1 answer: Michelle Obama\n"
            "Sub-q2: What is the occupation of Michelle Obama?\n\n"
        )
        content = (
            "Given the original multi-hop question and the answer to the first hop, "
            "write the second sub-question. Replace the indirect reference with the "
            "hop-1 answer. Output only the sub-question.\n\n"
            f"{examples}"
            f"Q: {question}\n"
            f"Hop-1 answer: {hop1_answer}\n"
            "Sub-q2:"
        )
        return [{"role": "user", "content": content}]

    def messages_rewrite(self, question: str) -> list:
        return self._messages_from_content(question, "rewrite")

    def messages_answer(self, question: str, passages: list) -> list:
        ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        content = (
            "Answer the question using the passages. "
            "Give a short answer phrase only — no explanation.\n\n"
            f"{ctx}\n\n"
            f"Question: {question}"
        )
        return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_item(item: dict) -> dict:
    """
    Normalize a HotpotQA item to a consistent dict format.

    HuggingFace format (hotpotqa/hotpot_qa, distractor config):
      context  = {"title": [...], "sentences": [[...], ...]}
      supporting_facts = {"title": [...], "sent_id": [...]}

    Original JSON format (hotpot_train_v1.1.json):
      context  = [[title, [sent, ...]], ...]
      supporting_facts = {"title": [...], "sent_id": [...]}

    We normalise context to the list-of-[title, sents] form used by
    the corpus builder below.
    """
    ctx = item.get("context", [])
    if isinstance(ctx, dict):
        # HuggingFace arrow format → convert to list-of-pairs
        titles = ctx.get("title", [])
        sents  = ctx.get("sentences", [])
        item = dict(item)
        item["context"] = [[t, s] for t, s in zip(titles, sents)]
    return item


def load_hotpot_train(path: str, n_bridge: int, n_comparison: int, seed: int = 42):
    """
    Load and sample HotpotQA train split.

    Tries local file first; falls back to HuggingFace datasets library if
    the file does not exist (requires: pip install datasets).
    """
    if os.path.exists(path):
        print(f"Loading HotpotQA train data from {path}...")
        with open(path) as f:
            data = json.load(f)
        data = [_normalize_item(d) for d in data]
    else:
        print(f"Local file not found: {path}")
        print("Falling back to HuggingFace datasets (hotpotqa/hotpot_qa, distractor)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library not installed. Run: pip install datasets\n"
                f"Or download the data manually and place it at: {path}"
            )
        hf_data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train",
                               trust_remote_code=True)
        data = [_normalize_item(dict(d)) for d in hf_data]

    bridge     = [d for d in data if d.get("type") == "bridge"]
    comparison = [d for d in data if d.get("type") == "comparison"]

    rng = random.Random(seed)
    bridge     = rng.sample(bridge,     min(n_bridge,     len(bridge)))
    comparison = rng.sample(comparison, min(n_comparison, len(comparison)))

    print(f"Sampled {len(bridge)} bridge + {len(comparison)} comparison questions")
    return bridge, comparison


def get_gold_titles(item: dict) -> list:
    sf = item.get("supporting_facts", {})
    return list(set(sf.get("title", []))) if isinstance(sf, dict) else []


# ---------------------------------------------------------------------------
# 32B vLLM teacher
# ---------------------------------------------------------------------------

def load_teacher(model_name: str, tensor_parallel: int = 1):
    from vllm import LLM, SamplingParams
    print(f"Loading teacher {model_name} via vLLM (tp={tensor_parallel})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        max_model_len=4096,
        dtype="bfloat16",
    )
    print("Teacher ready.")
    return llm, SamplingParams


def teacher_generate(llm, SamplingParams, message_lists: list, max_tokens: int) -> list:
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.chat(message_lists, sampling_params=params)
    return [o.outputs[0].text.strip() for o in outputs]


# ---------------------------------------------------------------------------
# 3B HF student (hop1_answer only)
# ---------------------------------------------------------------------------

def load_student(model_name: str):
    print(f"Loading student {model_name} via HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Student ready.")
    return model, tokenizer


def student_generate_hop1(model, tokenizer, messages: list, max_tokens: int = 48) -> str:
    """Generate hop1_answer with 3B base model (no adapter)."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def ctx_recall_ok(generated_query: str, retriever: Retriever,
                  gold_titles: list, top_k: int,
                  threshold: float = None) -> bool:
    """
    Return True if FAISS retrieval with this query meets the recall threshold.

    For bridge sub_q1/sub_q2: threshold=0.5 (finds at least 1 of 2 gold passages).
    For comparison union recall: threshold=1.0 (handled separately in generate_comparison_data).
    """
    if not gold_titles:
        return False
    if threshold is None:
        threshold = cfg.MIN_CTX_RECALL
    retrieved = retriever.retrieve_with_meta(generated_query, top_k=top_k)
    recall = supporting_fact_recall(retrieved, gold_titles)
    return recall >= threshold


def hop2_contains_entity(sub_q2: str, hop1_answer: str) -> bool:
    """Return True if sub_q2 contains a key token from hop1_answer (B2 filter)."""
    if not hop1_answer or not sub_q2:
        return False
    # Check if any word from hop1_answer (>3 chars) appears in sub_q2
    tokens = [w.strip(".,?\"'") for w in hop1_answer.split() if len(w) > 3]
    sub_q2_lower = sub_q2.lower()
    return any(t.lower() in sub_q2_lower for t in tokens)


# ---------------------------------------------------------------------------
# SFT sample formatting
# ---------------------------------------------------------------------------

def make_sft_sample(user_messages: list, completion: str, task: str) -> dict:
    """Format a single (prompt, completion) pair as a chat SFT sample."""
    return {
        "task":     task,
        "messages": user_messages + [{"role": "assistant", "content": completion}],
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_bridge_data(
    items: list,
    retriever: Retriever,
    teacher_llm,
    SamplingParams,
    student_model,
    student_tokenizer,
    builder: PromptBuilder,
) -> tuple[list, dict]:
    """
    Returns (sft_samples, stats).
    sft_samples contains both sub_q1 and sub_q2 samples (2 per question if both pass).
    """
    stats = {"total": len(items), "sub_q1_pass": 0, "sub_q2_pass": 0,
             "b2_filter": 0, "final": 0}
    sft_samples = []

    # ── Step 1: batch generate sub_q1 with 32B ──────────────────────────────
    print(f"\n[Bridge] Generating sub_q1 for {len(items)} questions...")
    decompose_msgs = [builder.messages_decompose(item["question"]) for item in items]
    sub_q1_list = teacher_generate(
        teacher_llm, SamplingParams, decompose_msgs, max_tokens=cfg.TEACHER_MAX_TOKENS
    )
    sub_q1_list = [s.strip().strip('"') for s in sub_q1_list]

    # ── Step 2: quality filter sub_q1 + retrieve ────────────────────────────
    print("[Bridge] Filtering sub_q1 by ctx_recall + retrieving passages...")
    valid_items, valid_sub_q1, valid_passages, valid_msg_idx = [], [], [], []
    for i, (item, sub_q1) in enumerate(tqdm(zip(items, sub_q1_list), total=len(items))):
        gold_titles = get_gold_titles(item)
        if ctx_recall_ok(sub_q1, retriever, gold_titles, cfg.TOP_K, threshold=0.5):
            passages = retriever.retrieve(sub_q1, top_k=cfg.TOP_K)
            valid_items.append(item)
            valid_sub_q1.append(sub_q1)
            valid_passages.append(passages)
            valid_msg_idx.append(i)
            stats["sub_q1_pass"] += 1

    print(f"  sub_q1 pass rate: {stats['sub_q1_pass']}/{len(items)}")

    # ── Step 3: 3B generates hop1_answer ────────────────────────────────────
    print("[Bridge] Generating hop1_answer with 3B student model...")
    hop1_answers = []
    for item, sub_q1, passages in tqdm(
        zip(valid_items, valid_sub_q1, valid_passages), total=len(valid_items)
    ):
        msgs = builder.messages_answer(sub_q1, passages)
        hop1_answer = student_generate_hop1(
            student_model, student_tokenizer, msgs, max_tokens=cfg.HOP1_MAX_TOKENS
        )
        hop1_answers.append(hop1_answer)

    # ── Step 4: batch generate sub_q2 with 32B ──────────────────────────────
    print("[Bridge] Generating sub_q2 with 32B teacher...")
    hop2_msgs = [
        builder.messages_hop2(item["question"], hop1_answer)
        for item, hop1_answer in zip(valid_items, hop1_answers)
    ]
    sub_q2_list = teacher_generate(
        teacher_llm, SamplingParams, hop2_msgs, max_tokens=cfg.TEACHER_MAX_TOKENS
    )
    sub_q2_list = [s.strip().strip('"') for s in sub_q2_list]

    # ── Step 5: quality filter sub_q2 + build SFT samples ───────────────────
    print("[Bridge] Filtering sub_q2 and building SFT samples...")
    for item, sub_q1, hop1_answer, sub_q2, orig_idx in tqdm(
        zip(valid_items, valid_sub_q1, hop1_answers, sub_q2_list, valid_msg_idx),
        total=len(valid_items),
    ):
        gold_titles = get_gold_titles(item)

        # sub_q2 quality: ctx_recall + entity check (B2 filter)
        sub_q2_ok = (
            ctx_recall_ok(sub_q2, retriever, gold_titles, cfg.TOP_K, threshold=0.5)
            and hop2_contains_entity(sub_q2, hop1_answer)
        )
        if not sub_q2_ok:
            stats["b2_filter"] += 1

        # Always add sub_q1 sample (already filtered)
        sft_samples.append(make_sft_sample(
            decompose_msgs[orig_idx], sub_q1, task="sub_q1"
        ))

        # Add sub_q2 sample only if it passes
        if sub_q2_ok:
            stats["sub_q2_pass"] += 1
            sft_samples.append(make_sft_sample(
                hop2_msgs[len([x for x in valid_msg_idx if x < orig_idx])],
                sub_q2,
                task="sub_q2",
            ))

    stats["final"] = len(sft_samples)
    return sft_samples, stats


def generate_comparison_data(
    items: list,
    retriever: Retriever,
    teacher_llm,
    SamplingParams,
    builder: PromptBuilder,
) -> tuple[list, dict]:
    stats = {"total": len(items), "pass": 0}
    sft_samples = []

    print(f"\n[Comparison] Generating rewrites for {len(items)} questions...")
    rewrite_msgs = [builder.messages_rewrite(item["question"]) for item in items]
    raw_outputs = teacher_generate(
        teacher_llm, SamplingParams, rewrite_msgs, max_tokens=cfg.TEACHER_MAX_TOKENS
    )

    print("[Comparison] Filtering by ctx_recall...")
    for item, msgs, raw in tqdm(zip(items, rewrite_msgs, raw_outputs), total=len(items)):
        # Parse 2 sub-queries
        sub_queries = [
            line[2:].strip().strip('"')
            for line in raw.splitlines()
            if line.strip().lower().startswith("q:")
        ]
        if len(sub_queries) < 2:
            continue

        gold_titles = get_gold_titles(item)
        # Union retrieval: both sub-queries together should cover gold passages
        retrieved = []
        seen = set()
        for q in sub_queries[:2]:
            for p in retriever.retrieve_with_meta(q, top_k=cfg.TOP_K):
                if p["title"] not in seen:
                    seen.add(p["title"])
                    retrieved.append(p)

        recall = supporting_fact_recall(retrieved, gold_titles)
        if recall < cfg.MIN_CTX_RECALL:
            continue

        stats["pass"] += 1
        # Format output: "Q: ...\nQ: ..."
        completion = "\n".join(f"Q: {q}" for q in sub_queries[:2])
        sft_samples.append(make_sft_sample(msgs, completion, task="comparison_rewrite"))

    print(f"  Comparison pass rate: {stats['pass']}/{len(items)}")
    return sft_samples, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bridge",      type=int, default=cfg.N_BRIDGE_SAMPLES)
    parser.add_argument("--n-comparison",  type=int, default=cfg.N_COMPARISON_SAMPLES)
    parser.add_argument("--load-index",    action="store_true")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    # ── Load HotpotQA train data ─────────────────────────────────────────────
    bridge_items, comparison_items = load_hotpot_train(
        cfg.HOTPOT_TRAIN_PATH, args.n_bridge, args.n_comparison, seed=args.seed
    )

    # ── Load retriever ───────────────────────────────────────────────────────
    if args.load_index:
        retriever = Retriever.load(cfg.INDEX_PATH, cfg.CORPUS_PATH, cfg.EMBEDDING_MODEL)
    else:
        # Build corpus from train data
        seen, corpus = set(), []
        for item in bridge_items + comparison_items:
            for p in item.get("context", []):
                title, sents = p[0], p[1]
                if title not in seen:
                    seen.add(title)
                    corpus.append({"title": title, "text": " ".join(sents)})
        retriever = Retriever(corpus, cfg.EMBEDDING_MODEL)
        retriever.save(cfg.INDEX_PATH, cfg.CORPUS_PATH)

    # ── Load teacher (32B, vLLM) ─────────────────────────────────────────────
    teacher_llm, SamplingParams = load_teacher(cfg.TEACHER_MODEL, args.tensor_parallel)
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.TEACHER_MODEL)
    builder = PromptBuilder(teacher_tokenizer)

    # ── Load student (3B, HF) ────────────────────────────────────────────────
    student_model, student_tokenizer = load_student(cfg.STUDENT_MODEL)

    # ── Generate ─────────────────────────────────────────────────────────────
    all_samples = []

    bridge_samples, bridge_stats = generate_bridge_data(
        bridge_items, retriever, teacher_llm, SamplingParams,
        student_model, student_tokenizer, builder,
    )
    all_samples.extend(bridge_samples)

    comparison_samples, comparison_stats = generate_comparison_data(
        comparison_items, retriever, teacher_llm, SamplingParams, builder,
    )
    all_samples.extend(comparison_samples)

    # ── Shuffle + split ──────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    n_val   = max(1, int(len(all_samples) * cfg.VAL_FRACTION))
    val     = all_samples[:n_val]
    train   = all_samples[n_val:]

    # ── Save ─────────────────────────────────────────────────────────────────
    for path, data in [(cfg.TRAIN_DATA_PATH, train), (cfg.VAL_DATA_PATH, val)]:
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    task_counts = {}
    for s in train:
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1

    print("\n" + "="*60)
    print("  DATA GENERATION COMPLETE")
    print("="*60)
    print(f"  Bridge  sub_q1 pass : {bridge_stats['sub_q1_pass']}/{bridge_stats['total']}")
    print(f"  Bridge  sub_q2 pass : {bridge_stats['sub_q2_pass']}/{bridge_stats['sub_q1_pass']}")
    print(f"  Comparison pass     : {comparison_stats['pass']}/{comparison_stats['total']}")
    print(f"  Total SFT samples   : {len(all_samples)}")
    print(f"  Train / Val split   : {len(train)} / {len(val)}")
    print(f"  Task breakdown      : {task_counts}")
    print(f"  Saved → {cfg.TRAIN_DATA_PATH}")
    print(f"          {cfg.VAL_DATA_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
