# training/utils/format.py
"""
Data formatting utilities shared across all training scripts.

Handles:
- HotpotQA distractor dataset format → internal passage format
- Building model input/output strings for SFT and GRPO
- Parsing model outputs back into structured fields
"""

from __future__ import annotations
import re
import random


# ---------------------------------------------------------------------------
# HotpotQA distractor format → internal format
# ---------------------------------------------------------------------------

def parse_hotpotqa_example(ex: dict) -> dict:
    """
    Convert a raw HotpotQA distractor example into our internal format.

    HotpotQA context schema:
        context = {
            "title":     ["Title1", "Title2", ...],       # 10 passages
            "sentences": [["sent1", "sent2"], [...], ...]  # sentences per passage
        }
        supporting_facts = {
            "title":   ["GoldTitle1", "GoldTitle2"],
            "sent_id": [0, 0]
        }
    """
    # Build passage list: join sentences into full text
    passages = []
    for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
        passages.append({
            "title": title,
            "text":  " ".join(sents).strip(),
        })

    gold_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))  # dedup, preserve order

    return {
        "id":             ex["id"],
        "question":       ex["question"].strip(),
        "answer":         ex["answer"].strip(),
        "type":           ex["type"],           # "bridge" | "comparison"
        "level":          ex["level"],
        "passages":       passages,             # list of {title, text}
        "gold_titles":    gold_titles,          # list of gold passage titles
    }


# ---------------------------------------------------------------------------
# Preprocessing: shuffle passages + optional gold removal
# ---------------------------------------------------------------------------

def preprocess_example(ex: dict, remove_gold: bool = False,
                        rng: random.Random = None,
                        distractor_pool: list | None = None) -> dict:
    """
    Apply preprocessing to a parsed example.

    Normal (remove_gold=False):
        Keep all 10 passages, shuffle order. Gold can appear anywhere.

    Gold-removed (remove_gold=True):
        Remove the 2 gold passages (leaving 8 distractors).
        Pad back to 10 by sampling 2 replacement passages from
        `distractor_pool` (passages from other examples in the dataset).
        This keeps all examples at exactly 10 passages — the model cannot
        use passage count as a shortcut to predict "insufficient context".
        Answer is set to "insufficient context".

    distractor_pool: list of {title, text} dicts from all examples.
                     Pre-built once in 01_prepare_dataset.py and reused.
                     If None, gold-removed examples will have 8 passages
                     (acceptable for small datasets / quick runs).
    """
    if rng is None:
        rng = random.Random()

    gold_set = set(ex["gold_titles"])
    passages = list(ex["passages"])

    if remove_gold:
        passages = [p for p in passages if p["title"] not in gold_set]
        # Pad back to 10 using pool passages not already present
        if distractor_pool:
            present = {p["title"] for p in passages}
            candidates = [p for p in distractor_pool if p["title"] not in present]
            n_needed = max(0, 10 - len(passages))
            if candidates and n_needed > 0:
                passages += rng.sample(candidates, min(n_needed, len(candidates)))
        answer = "insufficient context"
    else:
        answer = ex["answer"]

    rng.shuffle(passages)
    return {**ex, "passages": passages, "answer": answer, "gold_removed": remove_gold}


# ---------------------------------------------------------------------------
# Model input formatting
# ---------------------------------------------------------------------------

def format_context_block(passages: list[dict], max_passages: int = 10) -> str:
    """Format passage list as numbered context block."""
    lines = []
    for i, p in enumerate(passages[:max_passages], 1):
        lines.append(f"[{i}] {p['title']}\n{p['text']}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, passages: list[dict]) -> str:
    """Build the user-turn prompt used for both SFT and GRPO."""
    ctx = format_context_block(passages)
    return (
        f"<query>{question}</query>\n\n"
        f"<context>\n{ctx}\n</context>"
    )


def build_assistant_trace(ex: dict, sub_q1: str, intermediate: str,
                           sub_q2: str, final_answer: str,
                           qtype: str = "bridge") -> str:
    """
    Build the assistant-turn target string for SFT training.
    Works for both bridge and comparison question types.
    """
    if qtype == "bridge":
        think = (
            f"Type: bridge\n"
            f"Sub-question 1: {sub_q1}\n"
            f"Intermediate answer: {intermediate}\n"
            f"Sub-question 2: {sub_q2}"
        )
    else:
        # For comparison: sub_q1/sub_q2 are the two entity sub-queries
        # intermediate is "answer1 | answer2"
        parts = intermediate.split("|") if "|" in intermediate else [intermediate, ""]
        ans1, ans2 = parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        think = (
            f"Type: comparison\n"
            f"Sub-question 1: {sub_q1}\n"
            f"Intermediate answer 1: {ans1}\n"
            f"Sub-question 2: {sub_q2}\n"
            f"Intermediate answer 2: {ans2}"
        )

    return f"<think>\n{think}\n</think>\n<answer>{final_answer}</answer>"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract content from <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_think(text: str) -> str:
    """Extract content from <think>...</think> tags."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_sub_q1(think: str) -> str:
    m = re.search(r"Sub-question 1:\s*(.+?)(?:\n|$)", think)
    return m.group(1).strip() if m else ""


def extract_intermediate(think: str) -> str:
    """Extract first intermediate answer from think block."""
    m = re.search(r"Intermediate answer(?:\s*1)?:\s*(.+?)(?:\n|$)", think)
    return m.group(1).strip() if m else ""


def has_valid_format(text: str) -> bool:
    """Check that output has both <think> and <answer> tags."""
    return bool(
        re.search(r"<think>.*?</think>", text, re.DOTALL) and
        re.search(r"<answer>.*?</answer>", text, re.DOTALL)
    )


def is_circular_sub_q1(original: str, sub_q1: str, threshold: float = 0.7) -> bool:
    """Return True if sub_q1 is too similar to the original question (circular)."""
    def words(s):
        return set(re.findall(r'\b[a-z]{3,}\b', s.lower()))
    orig_w, sub_w = words(original), words(sub_q1)
    if not orig_w:
        return False
    overlap = len(orig_w & sub_w) / len(orig_w)
    return overlap >= threshold
