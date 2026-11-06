#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail CI if the manuscript's claim scope silently drifts back up.

The measurement-validity audit says the 660 labels are rule-classifier outputs
with no committed human adjudication. That conclusion is only useful if the
README and the LaTeX source keep saying so. This guard makes the downgrade a
machine-checked invariant instead of a one-time documentation edit.

    python scripts/check_claim_scope.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS = [
    "README.md",
    os.path.join("paper", "benign_rag_kbs.tex"),
    os.path.join("paper", "cover_letter.md"),
]

# Claims that are not supported by anything committed in this repository.
FORBIDDEN = [
    "95% human agreement",
    ">95\\% human agreement",
    "26 refusal patterns + 12 advice patterns",
    "26 refusal patterns (e.g.",
    "Knowledge-Based Systems (submitted)",
    "systematically overrides safety alignment in open-weight",
]

# What the docs must keep saying.
REQUIRED = {
    "README.md": [
        "exploratory",
        "rule",
        "not peer-reviewed",
        "measurement_validity",
    ],
    os.path.join("paper", "benign_rag_kbs.tex"): [
        "kappa",
        "measurement_validity",
        "not been human-adjudicated",
    ],
}


def read(rel):
    path = os.path.join(BASE, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    failures = []
    print("claim scope guard")

    with open(os.path.join(BASE, "results", "measurement_validity.json"),
              encoding="utf-8") as f:
        mv = json.load(f)
    human = mv.get("human_validation", {})
    kappa = human.get("cohens_kappa")
    expected = ("human_validated" if human.get("present") and kappa is not None
                and kappa >= 0.6 else "exploratory_rule_classifier_only")
    ok = mv.get("status") == expected
    print("  %s  validity status is %s (expected %s)" %
          ("PASS" if ok else "FAIL", mv.get("status"), expected))
    if not ok:
        failures.append("measurement_validity.json status / human kappa disagree")
    if not mv.get("claim_scope"):
        failures.append("measurement_validity.json has no claim_scope")
        print("  FAIL  measurement_validity.json has no claim_scope")

    for rel in DOCS:
        text = read(rel)
        if text is None:
            print("  SKIP  %s (not present)" % rel)
            continue
        low = text.lower()
        for phrase in FORBIDDEN:
            if phrase.lower() in low:
                failures.append("%s still contains unsupported claim: %r" % (rel, phrase))
                print("  FAIL  %s still contains %r" % (rel, phrase))
        for phrase in REQUIRED.get(rel, []):
            if phrase.lower() not in low:
                failures.append("%s is missing required scope wording: %r" % (rel, phrase))
                print("  FAIL  %s missing %r" % (rel, phrase))
        print("  CHECK %s scanned" % rel)

    if failures:
        print("\n%d claim-scope violation(s):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("\nPASS: no unsupported safety-rate or peer-review claims in the docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
