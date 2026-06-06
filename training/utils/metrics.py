# training/utils/metrics.py
"""
Metrics and GRPO reward functions.

Reward design:
  - Format gate (hard):   unparseable output → 0.0 for everything
  - Accuracy (0.70):      0.5*EM + 0.5*F1 on normalized answer
  - Grounding (0.25):     hop1_answer appears in gold supporting passage text
  - Length penalty:       soft penalty on verbose <answer> content (max -0.05)
"""

from __future__ import annotations
import re
import string
from collections import Counter

from .format import (
    extract_answer, extract_think, extract_sub_q1,
    extract_intermediate, has_valid_format,
)


# ---------------------------------------------------------------------------
# Text normalisation (SQuAD-style)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall    = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Grounding score
# ---------------------------------------------------------------------------

def _word_overlap(phrase: str, passage: str) -> float:
    """
    Soft grounding: fraction of content words in `phrase` that appear in `passage`.
    More robust than exact substring match — partial credit for paraphrases.
    Stopwords are excluded so short function words don't dominate.
    """
    _STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
             "is", "was", "are", "were", "be", "been", "by", "with"}
    p_words = [w for w in normalize_answer(phrase).split() if w not in _STOP]
    if not p_words:
        return 0.0
    doc_words = set(normalize_answer(passage).split())
    return sum(1 for w in p_words if w in doc_words) / len(p_words)


def grounding_score(think: str, passages: list[dict],
                    gold_titles: list[str], qtype: str) -> float:
    """
    Check whether intermediate answers are grounded in gold passages.

    Bridge:     word-overlap between hop1_answer and gold_titles[0] passage text.
    Comparison: average word-overlap for both intermediate answers.
    Uses soft overlap instead of exact substring match to reward paraphrases.
    """
    def passage_text(title: str) -> str:
        for p in passages:
            if p["title"] == title:
                return p["text"].lower()
        return ""

    if qtype == "bridge":
        hop1 = extract_intermediate(think)
        if not hop1 or not gold_titles:
            return 0.0
        gold_text = passage_text(gold_titles[0])
        return _word_overlap(hop1, gold_text)

    else:  # comparison
        m1 = re.search(r"Intermediate answer 1:\s*(.+?)(?:\n|$)", think)
        m2 = re.search(r"Intermediate answer 2:\s*(.+?)(?:\n|$)", think)
        ans1 = m1.group(1).strip() if m1 else ""
        ans2 = m2.group(1).strip() if m2 else ""
        scores = []
        for ans, title in zip([ans1, ans2], gold_titles[:2]):
            if ans and title:
                scores.append(_word_overlap(ans, passage_text(title)))
        return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Length penalty
# ---------------------------------------------------------------------------

def length_penalty(answer_text: str, max_tokens: int = 15,
                   max_penalty: float = 0.05) -> float:
    """
    Soft penalty if the answer is too verbose.
    Answers should be 1-5 words; penalise anything over max_tokens.
    """
    n = len(answer_text.split())
    if n <= max_tokens:
        return 0.0
    excess = min(n - max_tokens, 50)
    return -max_penalty * (excess / 50)


# ---------------------------------------------------------------------------
# Combined reward (used by GRPO trainer)
# ---------------------------------------------------------------------------

def compute_reward(completion: str, gold_answer: str,
                   passages: list[dict], gold_titles: list[str],
                   qtype: str,
                   accuracy_w: float = 0.70,
                   grounding_w: float = 0.25,
                   max_length_penalty: float = 0.05) -> float:
    """
    Full reward for one completion.
    Returns a float in roughly [-0.05, 1.0].
    """
    # Hard gate: unparseable → 0
    if not has_valid_format(completion):
        return 0.0

    pred  = extract_answer(completion)
    think = extract_think(completion)

    # 1. Accuracy
    em  = exact_match(pred, gold_answer)
    f1  = token_f1(pred, gold_answer)
    acc = 0.5 * em + 0.5 * f1

    # 2. Grounding
    gnd = grounding_score(think, passages, gold_titles, qtype)

    # 3. Length
    lp = length_penalty(pred, max_penalty=max_length_penalty)

    return accuracy_w * acc + grounding_w * gnd + lp


# ---------------------------------------------------------------------------
# GRPO reward functions (TRL GRPOTrainer interface)
# Each function receives: completions (list[str]), **batch_kwargs
# Batch kwargs are columns from the dataset passed through.
# ---------------------------------------------------------------------------

def reward_format(completions, **kwargs) -> list[float]:
    """Binary format gate — 1.0 if valid, 0.0 if not."""
    return [1.0 if has_valid_format(c) else 0.0 for c in completions]


def reward_accuracy(completions, gold_answers, **kwargs) -> list[float]:
    """0.5*EM + 0.5*F1 on normalized answer text."""
    rewards = []
    for c, gold in zip(completions, gold_answers):
        if not has_valid_format(c):
            rewards.append(0.0)
            continue
        pred = extract_answer(c)
        rewards.append(0.5 * exact_match(pred, gold) + 0.5 * token_f1(pred, gold))
    return rewards


def reward_grounding(completions, passages_json, gold_titles_json,
                     question_types, **kwargs) -> list[float]:
    """
    Intermediate answer grounding score.
    passages_json and gold_titles_json are JSON strings (dataset columns
    must be serialisable scalars for TRL DataLoader).
    """
    import json
    rewards = []
    for c, p_json, gt_json, qtype in zip(
        completions, passages_json, gold_titles_json, question_types
    ):
        if not has_valid_format(c):
            rewards.append(0.0)
            continue
        passages    = json.loads(p_json)
        gold_titles = json.loads(gt_json)
        think       = extract_think(c)
        rewards.append(grounding_score(think, passages, gold_titles, qtype))
    return rewards


def reward_length(completions, **kwargs) -> list[float]:
    """Small negative penalty for verbose answers."""
    rewards = []
    for c in completions:
        pred = extract_answer(c) if has_valid_format(c) else c
        rewards.append(length_penalty(pred))
    return rewards
