#!/usr/bin/env python3
"""
eval_upperbound.py — RAG-only upper bound using a large open-source model.

Evaluates a large model on the same HotpotQA benchmark using simple RAG:
retrieve the top-k passages for the original question and answer in one shot.
No classification, decomposition, or multi-hop reasoning — pure model + retrieval.

This gives a clean comparison point against the fine-tuned Qwen2.5-3B agent:

    Large model  + simple RAG   (this script)
    vs.
    Fine-tuned 3B + agentic pipeline  (run_eval.py)

Hardware guide (bfloat16, vLLM):
  Qwen/Qwen2.5-14B-Instruct        1× A100 40GB   (~28 GB)
  Qwen/Qwen2.5-32B-Instruct        1× A100 80GB   (~64 GB)
  Qwen/Qwen2.5-72B-Instruct        2× A100 80GB   (--tensor-parallel 2)
  meta-llama/Llama-3.3-70B-Instruct 2× A100 80GB  (--tensor-parallel 2)

Usage:
  pip install vllm
  python eval_upperbound.py --model Qwen/Qwen2.5-32B-Instruct
  python eval_upperbound.py --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel 2
  python eval_upperbound.py --model Qwen/Qwen2.5-14B-Instruct --n 100
  python eval_upperbound.py --model Qwen/Qwen2.5-32B-Instruct --load-index
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must be set before vLLM is imported — vLLM v1 forks its EngineCore process
# and CUDA cannot be re-initialized in a forked subprocess on Linux.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

import config
from src.retriever import Retriever
from src.evaluator import (
    exact_match, token_f1, supporting_fact_recall,
    answer_coverage, faithfulness, summarize,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_corpus(items):
    seen, corpus = set(), []
    for item in items:
        for p in item.get("paragraphs", []):
            if p["title"] not in seen:
                seen.add(p["title"])
                corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


# ---------------------------------------------------------------------------
# Model wrapper  (vLLM preferred, HuggingFace fallback)
# ---------------------------------------------------------------------------

class LargeModel:
    """
    Batched chat generation.
    vLLM is strongly preferred — it processes all prompts in parallel on GPU.
    HuggingFace fallback processes one prompt at a time (much slower).
    """

    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self._backend   = None
        self._load(tensor_parallel_size)

    def _load(self, tensor_parallel_size: int):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
            print(f"Loading {self.model_name} via vLLM "
                  f"(tensor_parallel={tensor_parallel_size})...")
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=8192,
                dtype="bfloat16",
            )
            self._SamplingParams = SamplingParams
            self._backend = "vllm"
            print("vLLM backend ready.")
        except ImportError:
            print("vLLM not installed — falling back to HuggingFace transformers (slow).")
            print("Install vLLM for GPU batch inference:  pip install vllm")
            self._load_hf(tensor_parallel_size)
        except Exception as e:
            print(f"vLLM failed to initialize ({e}).")
            print("Falling back to HuggingFace transformers...")
            self._load_hf(tensor_parallel_size)

    def _load_hf(self, _tensor_parallel_size: int):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._hf_tok = AutoTokenizer.from_pretrained(self.model_name)
        self._hf_mdl = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant,
            device_map="auto",
        )
        self._hf_mdl.eval()
        self._backend = "hf"
        print("HuggingFace backend ready.")

    def generate_batch(self, message_lists: list, max_tokens: int = 64) -> list:
        """
        message_lists : list of conversations, each is
                        [{"role": "user", "content": "..."}]
        Returns list of response strings, one per input.
        """
        if self._backend == "vllm":
            params  = self._SamplingParams(temperature=0, max_tokens=max_tokens)
            outputs = self._llm.chat(message_lists, sampling_params=params)
            return [o.outputs[0].text.strip() for o in outputs]
        # HF fallback — sequential
        return [
            self._hf_generate(msgs, max_tokens)
            for msgs in tqdm(message_lists, desc="  generating")
        ]

    def _hf_generate(self, messages: list, max_tokens: int) -> str:
        import torch
        prompt    = self._hf_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs    = self._hf_tok(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs    = {k: v.to(self._hf_mdl.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._hf_mdl.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._hf_tok.eos_token_id,
            )
        return self._hf_tok.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# RAG evaluation
# ---------------------------------------------------------------------------

def _make_prompt(question: str, passages: list) -> list:
    ctx = "\n\n".join(
        f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages)
    )
    content = (
        "Answer the question using the passages below. "
        "Give a short answer phrase only — no explanation.\n\n"
        f"{ctx}\n\nQuestion: {question}"
    )
    return [{"role": "user", "content": content}]


def run_rag(items: list, retriever: Retriever, model: LargeModel, top_k: int) -> list:
    """
    Simple RAG: retrieve top-k passages per question, answer in one shot.
    No classification, decomposition, or multi-hop reasoning.
    """
    questions = [item["question"] for item in items]

    print(f"\nRetrieving passages for {len(items)} questions...")
    retrieved_all = [
        retriever.retrieve_with_meta(q, top_k=top_k)
        for q in tqdm(questions, desc="  retrieval")
    ]

    print(f"Generating answers ({len(items)} questions)...")
    prompts     = [_make_prompt(q, p) for q, p in zip(questions, retrieved_all)]
    predictions = model.generate_batch(prompts, max_tokens=64)

    results = []
    for item, pred, retrieved in zip(items, predictions, retrieved_all):
        sf          = item.get("supporting_facts", {})
        gold_titles = list(set(sf.get("title", []))) if isinstance(sf, dict) else []
        pred        = pred.strip()
        results.append({
            "id":             item["id"],
            "question":       item["question"],
            "gold":           item["answer"],
            "prediction":     pred,
            "type":           item.get("type"),
            "level":          item.get("level"),
            "retrieval_mode": "rag",
            "em":             exact_match(pred, item["answer"]),
            "f1":             token_f1(pred, item["answer"]),
            "ctx_recall":     supporting_fact_recall(retrieved, gold_titles),
            "ctx_precision":  None,
            "ans_coverage":   answer_coverage(retrieved, item["answer"]),
            "faithfulness":   faithfulness(pred, retrieved),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="Qwen/Qwen2.5-32B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--eval",            default=config.EVAL_PATH)
    parser.add_argument("--top-k",           type=int, default=config.TOP_K)
    parser.add_argument("--n",               type=int, default=None,
                        help="Evaluate only first N questions")
    parser.add_argument("--load-index",      action="store_true",
                        help="Load pre-built FAISS index from config.INDEX_PATH")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (vLLM only)")
    parser.add_argument("--output",          default=None,
                        help="Output JSON path (auto-named if omitted)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    eval_items = load_jsonl(args.eval)
    if args.n:
        eval_items = eval_items[:args.n]
    print(f"Loaded {len(eval_items)} eval items from {args.eval}")

    if args.output is None:
        slug = args.model.replace("/", "_").replace("-", "_").lower()
        args.output = os.path.join(RESULTS_DIR, f"eval_{slug}_n{len(eval_items)}_rag.json")

    # Load retriever
    if args.load_index:
        retriever = Retriever.load(config.INDEX_PATH, config.CORPUS_PATH, config.EMBEDDING_MODEL)
    else:
        corpus    = build_corpus(eval_items)
        retriever = Retriever(corpus, config.EMBEDDING_MODEL)
        retriever.save(config.INDEX_PATH, config.CORPUS_PATH)

    # Load model
    model = LargeModel(args.model, tensor_parallel_size=args.tensor_parallel)

    # Run eval
    results = run_rag(eval_items, retriever, model, args.top_k)

    # Summary
    summary = summarize(results)
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  UPPER BOUND (RAG-only)  |  {args.model}")
    print(f"  n={summary['n']}  top-k={args.top_k}")
    print(sep)
    print(f"  Exact Match    : {summary['em']:.4f}   (agent baseline: 0.5600)")
    print(f"  Token F1       : {summary['f1']:.4f}   (agent baseline: 0.6751)")
    print(f"  Context Recall : {summary['ctx_recall']:.4f}")
    print(f"  Ans Coverage   : {summary['ans_coverage']:.4f}")
    print(f"\n  By question type:")
    for qtype, s in summary["by_type"].items():
        print(f"    {qtype:<14} EM={s['em']:.4f}  F1={s['f1']:.4f}  (n={s['count']})")
    print(f"\n  By difficulty:")
    for level, s in summary["by_level"].items():
        print(f"    {level:<14} EM={s['em']:.4f}  F1={s['f1']:.4f}  (n={s['count']})")
    print(sep)

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_item": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
