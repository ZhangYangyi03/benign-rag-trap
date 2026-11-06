# Cover letter — DRAFT, NOT SENT

> This file is a **draft kept in the repository for completeness**. It has not been sent to
> any journal, the manuscript has not been peer-reviewed, and no submission or editorial
> decision is claimed. It is reproduced here with the claims scoped to what the committed
> data actually supports. See `README.md` → *Measurement validity*.

---

Dear Editor,

I am writing to enquire whether a manuscript entitled *"The Benign RAG Trap: Benign Deployment Framing and Family-Dependent Refusal Behaviour in Open-Weight LLMs"* would be within the scope of Knowledge-Based Systems.

**Why this might fit KBS.**

The work treats the system prompt and the RAG delivery pipeline as a knowledge-delivery configuration, and asks whether that configuration is a neutral channel or a safety-relevant variable. We measure it on three open-weight model families rather than asserting it from a single model.

**What the manuscript reports, and what it does not.**

1. A controlled measurement of benign framing effects across three open-weight LLM families (660 trials). Under a deterministic substring classifier, the rule-classified safe-refusal rate orders as 96.8% / 58.6% / 23.6%. That ordering survives seven labelling perturbations and both bootstrap 95% confidence intervals exclude zero.

2. A **limitations-first statement of measurement validity**: the 660 labels have **not been human-adjudicated**. A related audit by the same author found a rule classifier used as ground truth was correct on 53% of hand-labelled items (kappa 0.115). We therefore report the rates as classifier outputs, not as human-judged safety rates, and ship `scripts/measurement_validity.py`, a 60-item human-labelling template, and a CI guard that keeps the claim scope downgraded until Cohen's kappa reaches 0.6.

3. An activation analysis (9 layers × 2 modules × 3 models) that is **correlational**. Hidden-state shifts are reported as measurements; the causal interpretation is explicitly a hypothesis.

4. All raw responses, the exact classifier, the sensitivity analysis, and the bootstrap code are committed so a reader can re-derive every quoted number without the original hardware.

**Known gaps, stated up front:** no human adjudication of the behavioural labels; single-sample decoding at temperature 0.7; three 7–8B families on one local machine; no external replication; no peer review. We would regard human validation of the labeler as a prerequisite before any of these figures are used to characterise model safety.

**Reproducibility.** All code and experimental data are available at
<https://github.com/ZhangYangyi03/benign-rag-trap>. The analysis path uses only the Python standard library.

Sincerely,
Yangyi Zhang
Georgetown University
