#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-derive the headline numbers in README.md from results/ and check them.

Every number this paper puts in its abstract is recomputed here from the
raw trial JSONs. If a number in the README stops matching the data, this
script fails and says which one. Standard library only.

    python scripts/verify_readme_numbers.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")


def load(*parts):
    with open(os.path.join(RES, *parts), encoding="utf-8") as f:
        return json.load(f)


def share(records, label):
    return 100.0 * sum(1 for r in records if r["classification"] == label) / len(records)


def chi2_2x2(a, b, c, d):
    """Pearson chi-square for a 2x2 table [[a,b],[c,d]], 1 dof."""
    n = a + b + c + d
    row1, row2, col1, col2 = a + b, c + d, a + c, b + d
    if min(row1, row2, col1, col2) == 0:
        return 0.0
    e = (row1 * col1 / n, row1 * col2 / n, row2 * col1 / n, row2 * col2 / n)
    return sum((o - x) ** 2 / x for o, x in zip((a, b, c, d), e) if x > 0)


def main():
    llama = load("cross_family", "llama3_8b_framings.json")
    qwen = load("benign_rag_expanded_results.json")
    mistral = load("cross_family", "mistral_7b_framings.json")

    checks = []

    def check(name, got, want, tol=0.05):
        ok = abs(got - want) <= tol
        checks.append((ok, name, got, want))
        print(("  PASS  " if ok else "  FAIL  ") +
              f"{name:52s} got={got:8.2f} readme={want:8.2f}")

    print("Safe-refusal rate per family (README: 96.8 / 58.6 / 23.6)")
    check("Llama-3.1-8B safe-refusal %", share(llama, "SAFE_REFUSAL"), 96.8)
    check("Qwen2.5-7B safe-refusal %", share(qwen, "SAFE_REFUSAL"), 58.6)
    check("Mistral-7B safe-refusal %", share(mistral, "SAFE_REFUSAL"), 23.6)

    print("Loss modality (README: Qwen unsafe 18.2 / Mistral ambiguous 56.8)")
    check("Qwen2.5-7B unsafe %", share(qwen, "UNSAFE"), 18.2)
    check("Mistral-7B ambiguous %", share(mistral, "AMBIGUOUS"), 56.8)

    print("Trial counts (README: 22 prompts x 10 framings x 3 families)")
    for name, rs in (("llama3", llama), ("qwen", qwen), ("mistral", mistral)):
        checks.append((len(rs) == 220, f"{name} trial count", len(rs), 220))
        print(("  PASS  " if len(rs) == 220 else "  FAIL  ") +
              f"{name + ' trial count':52s} got={len(rs):8d} readme={220:8d}")

    print("Statistics (README: chi2=51.9 p<1e-12 for the ambiguity split;")
    print("            chi2=0.1 p=0.72 for unsafe Qwen-vs-Mistral)")
    q_amb = sum(1 for r in qwen if r["classification"] == "AMBIGUOUS")
    m_amb = sum(1 for r in mistral if r["classification"] == "AMBIGUOUS")
    chi_amb = chi2_2x2(q_amb, len(qwen) - q_amb, m_amb, len(mistral) - m_amb)
    check("chi2 Qwen-vs-Mistral ambiguous", chi_amb, 51.9, tol=0.2)

    q_un = sum(1 for r in qwen if r["classification"] == "UNSAFE")
    m_un = sum(1 for r in mistral if r["classification"] == "UNSAFE")
    chi_un = chi2_2x2(q_un, len(qwen) - q_un, m_un, len(mistral) - m_un)
    check("chi2 Qwen-vs-Mistral unsafe", chi_un, 0.1, tol=0.2)

    print("Mentor framing (README: 36% unsafe across both vulnerable families)")
    for name, rs in (("qwen", qwen), ("mistral", mistral)):
        mentor = [r for r in rs if r["config"] == "mentor"]
        check(f"{name} mentor unsafe %", share(mentor, "UNSAFE"), 36.4, tol=0.2)

    bad = [c for c in checks if not c[0]]
    print()
    if bad:
        print(f"{len(bad)} of {len(checks)} README numbers do NOT match results/")
        return 1
    print(f"all {len(checks)} README numbers reproduce from results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
