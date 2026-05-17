#!/usr/bin/env python3
"""
analyze_results.py — inspect failures in eval_live.json.

Usage:
    python3 analyze_results.py                        # full analysis
    python3 analyze_results.py --type bridge          # bridge only
    python3 analyze_results.py --type comparison      # comparison only
    python3 analyze_results.py --show-failures 20     # print first 20 failure cases
"""

import argparse
import json
import os
from collections import Counter

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "eval_live.json")


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["per_item"], data["summary"]


def bucket(item):
    """Classify a result into a failure/success category."""
    em   = item["em"]
    f1   = item["f1"]
    mode = item.get("retrieval_mode", "")
    if em == 1:
        return "correct"
    if f1 >= 0.5:
        return "partial"       # right idea, wrong surface form
    if item.get("ctx_recall", 0) == 0:
        return "retrieval_miss" # gold passage never retrieved
    return "model_fail"         # passages were there, model got it wrong


def print_section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",       default=RESULTS_PATH)
    parser.add_argument("--type",          choices=["bridge", "comparison"], default=None)
    parser.add_argument("--show-failures", type=int, default=15,
                        help="Number of failure cases to print in detail")
    args = parser.parse_args()

    items, summary = load_results(args.results)

    if args.type:
        items = [i for i in items if i.get("type") == args.type]

    print(f"\nAnalysing {len(items)} items"
          + (f" (type={args.type})" if args.type else ""))

    # ── Overall breakdown ─────────────────────────────────────────────────
    print_section("Outcome breakdown")
    counts = Counter(bucket(i) for i in items)
    total  = len(items)
    for cat in ["correct", "partial", "retrieval_miss", "model_fail"]:
        n = counts[cat]
        print(f"  {cat:<20} {n:>4}  ({100*n/total:.1f}%)")

    # ── Bridge-specific: decomposition quality ────────────────────────────
    bridge_items = [i for i in items if i.get("type") == "bridge"]
    if bridge_items:
        print_section("Bridge — decomposition analysis")

        no_sub_q1   = sum(1 for i in bridge_items if not i.get("sub_q1"))
        no_hop1_ans = sum(1 for i in bridge_items if not i.get("hop1_answer"))
        no_sub_q2   = sum(1 for i in bridge_items if not i.get("sub_q2"))
        web_used    = sum(1 for i in bridge_items if i.get("retrieval_mode") == "web_search")

        print(f"  sub_q1 missing       : {no_sub_q1}")
        print(f"  hop1_answer missing  : {no_hop1_ans}")
        print(f"  sub_q2 missing       : {no_sub_q2}")
        print(f"  web search used      : {web_used}")

        # EM by retrieval mode
        print_section("Bridge — EM by retrieval mode")
        modes = Counter(i.get("retrieval_mode") for i in bridge_items)
        for mode, count in modes.most_common():
            subset = [i for i in bridge_items if i.get("retrieval_mode") == mode]
            em = sum(i["em"] for i in subset) / len(subset)
            f1 = sum(i["f1"] for i in subset) / len(subset)
            print(f"  {mode:<25} n={count:<4} EM={em:.3f}  F1={f1:.3f}")

    # ── Comparison-specific ───────────────────────────────────────────────
    comp_items = [i for i in items if i.get("type") == "comparison"]
    if comp_items:
        print_section("Comparison — EM by retrieval mode")
        modes = Counter(i.get("retrieval_mode") for i in comp_items)
        for mode, count in modes.most_common():
            subset = [i for i in comp_items if i.get("retrieval_mode") == mode]
            em = sum(i["em"] for i in subset) / len(subset)
            f1 = sum(i["f1"] for i in subset) / len(subset)
            print(f"  {mode:<25} n={count:<4} EM={em:.3f}  F1={f1:.3f}")

    # ── Retry / web search impact ─────────────────────────────────────────
    retried = [i for i in items if i.get("retry_count", 0) > 0]
    if retried:
        print_section("Retry / web search impact")
        em_retried = sum(i["em"] for i in retried) / len(retried)
        em_no_retry = sum(i["em"] for i in items if i.get("retry_count", 0) == 0)
        em_no_retry /= max(1, sum(1 for i in items if i.get("retry_count", 0) == 0))
        print(f"  Questions retried      : {len(retried)}")
        print(f"  EM on retried          : {em_retried:.3f}")
        print(f"  EM on non-retried      : {em_no_retry:.3f}")

    # ── Failure case details ──────────────────────────────────────────────
    failures = [i for i in items if i["em"] == 0]
    failures.sort(key=lambda x: x["f1"])  # worst first

    if args.type:
        section_label = f"Failure cases ({args.type}, worst F1 first)"
    else:
        section_label = "Failure cases (worst F1 first)"

    print_section(f"{section_label} — showing {min(args.show_failures, len(failures))}/{len(failures)}")

    for item in failures[:args.show_failures]:
        print(f"\n  Q   : {item['question']}")
        print(f"  Gold: {item['gold']}")
        print(f"  Pred: {item['prediction']}")
        print(f"  F1={item['f1']:.2f}  mode={item.get('retrieval_mode')}  retries={item.get('retry_count',0)}")
        if item.get("type") == "bridge":
            print(f"  sub_q1     : {item.get('sub_q1')}")
            print(f"  hop1_answer: {item.get('hop1_answer')}")
            print(f"  sub_q2     : {item.get('sub_q2')}")
        elif item.get("type") == "comparison":
            print(f"  queries: {item.get('queries_used')}")


if __name__ == "__main__":
    main()
