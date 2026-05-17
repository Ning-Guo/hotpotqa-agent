# src/evaluator.py
import re
import string
from collections import Counter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Answer normalisation (SQuAD-style)
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_fact_recall(retrieved_paragraphs, gold_titles):
    """
    Context Recall — fraction of gold paragraph titles that were retrieved.
    Measures whether the retriever found all the evidence needed.

    Example:
      gold_titles     = ["Arthur's Magazine", "First for Women"]
      retrieved titles = ["Arthur's Magazine", "First for Women", "Radio City"]
      → recall = 2/2 = 1.0
    """
    if not gold_titles:
        return 0.0
    retrieved_titles = {p["title"] for p in retrieved_paragraphs}
    return len(set(gold_titles) & retrieved_titles) / len(set(gold_titles))


def context_precision(retrieved_paragraphs):
    """
    Context Precision — fraction of retrieved paragraphs that are gold.
    Measures retriever signal-to-noise ratio.

    Example:
      retrieved = [gold, gold, distractor]
      → precision = 2/3 = 0.67
    """
    if not retrieved_paragraphs:
        return 0.0
    gold_count = sum(1 for p in retrieved_paragraphs if p.get("is_gold", False))
    return gold_count / len(retrieved_paragraphs)


def answer_coverage(retrieved_paragraphs, gold_answer):
    """
    Answer Coverage — whether the gold answer string appears anywhere in the
    retrieved context (after normalisation).  Binary: 1 or 0.

    Tells you: even if the model answers wrong, did the retriever at least
    surface the right text?

    Example:
      gold_answer = "Arthur's Magazine"  →  "arthurs magazine"
      context contains "arthurs magazine was an american..."  →  1
    """
    context = " ".join(
        normalize_answer(f"{p['title']} {p['text']}") for p in retrieved_paragraphs
    )
    return int(normalize_answer(gold_answer) in context)


def faithfulness(prediction, retrieved_paragraphs):
    """
    Faithfulness — fraction of predicted answer tokens that appear in the
    retrieved context.  Measures whether the answer is grounded in what was
    retrieved vs. hallucinated from model memory.

    Example:
      prediction = "Arthur's Magazine"   →  ["arthurs", "magazine"]
      context contains both tokens        →  faithfulness = 1.0

      prediction = "Arthur founded it"   →  ["arthur", "founded", "it"]
      context has "arthur" but not "founded" or "it" as content words
      →  faithfulness = 1/3 ≈ 0.33
    """
    if not retrieved_paragraphs or not prediction.strip():
        return 0.0
    context_tokens = set(
        normalize_answer(" ".join(p["text"] for p in retrieved_paragraphs)).split()
    )
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    grounded = sum(1 for t in pred_tokens if t in context_tokens)
    return grounded / len(pred_tokens)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(agent, dataset, retriever=None, top_k=3):
    """
    Runs the agent over the dataset and returns a list of per-item result dicts.

    When hallucination_detector is provided (and retriever is active), each row
    also gets: entailment, contradiction, neutral, outcome.
    """
    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        row = {
            "id":       item["id"],
            "question": item["question"],
            "gold":     item["answer"],
            "type":     item["type"],
            "level":    item["level"],
        }

        if retriever:
            retrieved = retriever.retrieve_with_meta(item["question"], top_k=top_k)
            retrieved_texts = [f"{p['title']}\n{p['text']}" for p in retrieved]
            row["ctx_recall"]    = supporting_fact_recall(retrieved, item["supporting_facts"]["title"])
            row["ctx_precision"] = context_precision(retrieved)
            row["ans_coverage"]  = answer_coverage(retrieved, item["answer"])
        else:
            retrieved         = None
            retrieved_texts   = None
            row["ctx_recall"]    = None
            row["ctx_precision"] = None
            row["ans_coverage"]  = None

        row["prediction"]   = agent.answer(item["question"], retrieved_texts)
        row["em"]           = exact_match(row["prediction"], row["gold"])
        row["f1"]           = token_f1(row["prediction"], row["gold"])
        row["faithfulness"] = faithfulness(row["prediction"], retrieved) if retrieved else None

        row["entailment"]    = None
        row["contradiction"] = None
        row["neutral"]       = None
        row["outcome"]       = None
        results.append(row)

    return results


def summarize(results):
    """Aggregates per-item results into a summary dict."""

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def outcome_counts(rows):
        cats = ["grounded_correct", "grounded_wrong", "parametric_correct", "hallucinated"]
        total = sum(1 for r in rows if r["outcome"] is not None)
        if total == 0:
            return None
        return {
            c: round(sum(1 for r in rows if r["outcome"] == c) / total, 4)
            for c in cats
        }

    summary = {
        # Generation quality
        "em": mean([r["em"] for r in results]),
        "f1": mean([r["f1"] for r in results]),
        # Retriever quality  (None when running baseline)
        "ctx_recall":    mean([r["ctx_recall"]    for r in results]),
        "ctx_precision": mean([r["ctx_precision"] for r in results]),
        "ans_coverage":  mean([r["ans_coverage"]  for r in results]),
        # RAG faithfulness (token-overlap proxy)
        "faithfulness":  mean([r["faithfulness"]  for r in results]),
        # NLI-based hallucination scores
        "entailment":    mean([r["entailment"]    for r in results]),
        "contradiction": mean([r["contradiction"] for r in results]),
        # Outcome breakdown (only present when NLI was run)
        "outcomes": outcome_counts(results),
        "n": len(results),
        "by_type":  {},
        "by_level": {},
    }

    for dim, key in [("by_type", "type"), ("by_level", "level")]:
        for val in sorted({r[key] for r in results}):
            subset = [r for r in results if r[key] == val]
            summary[dim][val] = {
                "em":    mean([r["em"] for r in subset]),
                "f1":    mean([r["f1"] for r in subset]),
                "count": len(subset),
            }

    return summary
