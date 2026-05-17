# src/reasoner.py
"""
Live inference-time reasoning using the base Qwen2.5-3B-Instruct model
(adapter disabled via PeftModel.disable_adapter()).

Handles all reasoning steps that were pre-computed offline in the prior project:
  - question type classification  (bridge vs comparison)
  - bridge decomposition          (sub_q1 generation)
  - hop-1 answering               (intermediate answer for bridge)
  - hop-2 query formulation       (sub_q2 with hop1_answer inserted)
  - comparison query rewriting    (union sub-queries)
"""

from __future__ import annotations

import re
import torch


class Reasoner:
    """
    All generation runs with the LoRA adapter disabled so we get the base
    model's unbiased reasoning rather than the GRPO answer-synthesis style.
    """

    def __init__(self, model, tokenizer, device: str, max_new_tokens: int = 48):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, question: str) -> str:
        """Returns 'bridge' or 'comparison'."""
        # Keyword heuristic: questions comparing two explicitly named entities
        # are almost always comparison. This catches patterns the small model
        # frequently misreads as bridge (e.g. "Were X and Y of the same X?").
        if _looks_like_comparison(question):
            return "comparison"
        prompt = self._classify_prompt(question)
        raw    = self._generate(prompt, max_new_tokens=8)
        return "comparison" if "comparison" in raw.lower() else "bridge"

    def decompose_bridge(self, question: str) -> str:
        """Returns sub_q1: the first-hop sub-question."""
        prompt = self._decompose_prompt(question)
        return self._generate(prompt).strip().strip('"')

    def answer_hop1(self, sub_q1: str, passages: list[str]) -> str:
        """Short answer for sub_q1 using retrieved passages (intermediate hop)."""
        prompt = self._answer_prompt(sub_q1, passages)
        return self._generate(prompt).strip()

    def formulate_hop2(self, original_question: str, hop1_answer: str) -> str:
        """Returns sub_q2 with hop1_answer substituted in."""
        prompt = self._hop2_prompt(original_question, hop1_answer)
        return self._generate(prompt).strip().strip('"')

    def rewrite_comparison(self, question: str) -> list[str]:
        """
        Returns a list of 2 simple lookup queries for union retrieval.
        Falls back to the original question if parsing fails.
        """
        prompt = self._rewrite_prompt(question)
        raw    = self._generate(prompt, max_new_tokens=80)
        queries = self._parse_rewritten_queries(raw)
        return queries if len(queries) >= 2 else [question]

    # ------------------------------------------------------------------
    # Prompt templates (few-shot, proven effective on Qwen2.5-3B)
    # ------------------------------------------------------------------

    def _classify_prompt(self, question: str) -> str:
        examples = (
            'Q: "Who was the director of the film in which Julia Roberts starred in 1990?" → bridge\n'
            'Q: "Which country has a larger population, China or India?" → comparison\n'
            'Q: "What is the capital of the country where Einstein was born?" → bridge\n'
            'Q: "Which was published first, Animal Farm or Brave New World?" → comparison\n'
            'Q: "Were Scott Derrickson and Ed Wood of the same nationality?" → comparison\n'
            'Q: "Are both Variety and The Hollywood Reporter entertainment publications?" → comparison\n'
            'Q: "What government position was held by the person who took over from Simon Rattle?" → bridge\n'
            'Q: "Do David Copperfield and Hamlet both have the same author?" → comparison\n'
            'Q: "What country of origin does Flying Tigers and Claire Lee Chennault have in common?" → comparison\n'
            'Q: "Which Academy Award-winning actress starred with Alec Baldwin in both the Broadway revival and TV movie of A Streetcar Named Desire?" → bridge\n'
            'Q: "Lake Hodges is crossed by the major Interstate Highway that begins in what county?" → bridge\n'
        )
        content = (
            "Classify each question as 'bridge' or 'comparison'.\n"
            "bridge: the question refers to an entity indirectly (via a film, role, or relationship) "
            "and asks about a property of that entity — you must find the entity first.\n"
            "comparison: two or more entities are explicitly named by name, and a single property "
            "is compared or verified across them (same, both, which is older/larger/first, etc.)\n\n"
            f"{examples}"
            f'Q: "{question}" →'
        )
        return self._chat(content)

    def _decompose_prompt(self, question: str) -> str:
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
        return self._chat(content)

    def _answer_prompt(self, question: str, passages: list[str]) -> str:
        ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        content = (
            "Answer the question using the passages. "
            "Give a short answer phrase only — no explanation.\n\n"
            f"{ctx}\n\n"
            f"Question: {question}"
        )
        return self._chat(content)

    def _hop2_prompt(self, original_question: str, hop1_answer: str) -> str:
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
            f"Q: {original_question}\n"
            f"Hop-1 answer: {hop1_answer}\n"
            "Sub-q2:"
        )
        return self._chat(content)

    def _rewrite_prompt(self, question: str) -> str:
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
        return self._chat(content)

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _chat(self, user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompt: str, max_new_tokens=None) -> str:
        n_tokens = max_new_tokens or self.max_new_tokens
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Disable LoRA adapter — use base model for reasoning
            with self.model.disable_adapter():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=n_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _parse_rewritten_queries(self, raw: str) -> list[str]:
        """Extract lines starting with 'Q:' from the model output."""
        queries = []
        for line in raw.splitlines():
            line = line.strip()
            if line.lower().startswith("q:"):
                q = line[2:].strip().strip('"')
                if q:
                    queries.append(q)
        return queries


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _looks_like_comparison(question: str) -> bool:
    """
    Keyword heuristic for comparison questions. Fires when the question
    contains comparison signal words AND at least two capitalised names,
    catching patterns like "Were X and Y of the same nationality?" that
    Qwen2.5-3B frequently misclassifies as bridge.
    """
    q = question.lower()

    comparison_signals = [
        "same", "either", "neither", "in common", "have in common",
        "which is older", "which is newer", "which is larger", "which is smaller",
        "which is longer", "which is shorter", "which is taller",
        "which came first", "which was first", "which was founded",
        "which was born", "which is more", "which is less",
        "were both", "are both", "did both", "do both",
        "were either", "are either",
    ]
    if not any(sig in q for sig in comparison_signals):
        return False

    # Require at least two capitalised tokens (proper nouns / named entities)
    capitalised = re.findall(r'\b[A-Z][a-z]+\b', question)
    return len(capitalised) >= 2
