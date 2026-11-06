# The Benign RAG Trap

**Benign Deployment Framing and Family-Dependent Refusal Behaviour in Open-Weight LLMs**

> **Status: exploratory, not peer-reviewed.** This repository holds 660 stored model
> responses, a deterministic rule labeler, a LaTeX manuscript draft and a cover-letter
> draft. No external review, no external reproduction, and **no human adjudication of the
> 660 labels** has taken place. Every percentage below describes *what the rule labeler
> counted*, not a human-judged safety rate. See
> [`results/measurement_validity.json`](results/measurement_validity.json) and
> [Measurement validity](#measurement-validity) before quoting any number.

This repository contains the full experimental data, analysis code, and manuscript draft for a study of benign system-prompt framing as a safety-degradation vector in open-weight LLMs.

## Paper

- **Manuscript**: `paper/benign_rag_kbs.pdf` (KBS/Elsevier *format*, 15 pages, 6 figures) — the format is a typesetting choice, **not** a statement of acceptance or review.
- **LaTeX source**: `paper/benign_rag_kbs.tex`
- **References**: `paper/benign_rag_kbs.bib` (15 entries, elsarticle-harv style)
- **Cover letter**: `paper/cover_letter.md` — **a draft that has not been sent**

The manuscript and cover letter are unpublished working documents. They are **not peer-reviewed** and no journal submission or decision is claimed here.

## Key findings (rule-labeler outputs)

1. **Family-dependent refusal ordering**: rule-classified safe-refusal rates under 10 benign role framings are 96.8% (Llama-3.1-8B), 58.6% (Qwen2.5-7B) and 23.6% (Mistral-7B) — a 4× spread. The *ordering* survives all seven labelling perturbations and both bootstrap 95% CIs exclude zero (see below). The *absolute levels* do not: see the caveat.

2. **Loss modality is family-specific under the baseline labeler**: Qwen converts refusal into rule-classified unsafe content (18.2% unsafe); Mistral converts it into long-form ambiguous analysis (56.8% ambiguous, χ²=51.9, p<10⁻¹²). The two vulnerable families have statistically indistinguishable unsafe rates (χ²=0.1, p=0.72).

3. **Mentor framing**: 36% rule-classified unsafe output across both vulnerable families.

4. **Mechanism — attention-layer representation shift**: benign framing shifts attention-layer hidden states in vulnerable families (Qwen mean delta-norm 0.032) while the immune family's attention layers stay nearly unchanged (Llama-3.1: 0.009, 3.75× difference). MLP shifts are large across all families and do not distinguish immune vs. vulnerable. Hidden-state data is included; the causal interpretation in the manuscript is a hypothesis, not an established mechanism.

## Measurement validity

An independent audit by the same author
([`llm-safety-eval-audits`](https://github.com/ZhangYangyi03/llm-safety-eval-audits)) found that a rule classifier treated as ground truth was correct on only **53%** of hand-labelled items, with **kappa = 0.115** against an LLM judge. This repository used the same kind of labeler. Rather than leave that as a disclaimer, `scripts/measurement_validity.py` makes it measurable and CI enforces the result:

| Check | Result |
|---|---|
| Committed labels replayed from committed response text | **660/660 (100%)** — the labels are internally consistent, but that only proves the code ran |
| Family ordering `L > Q > M` under 7 pattern/threshold perturbations | **7/7 hold** |
| Ambiguity split Qwen vs Mistral (χ² > 10.83) under those 7 | **7/7 hold** |
| Bootstrap 95% CI, Llama − Mistral refusal gap | **+73.18 pp [+66.82, +79.55]** |
| Bootstrap 95% CI, Qwen − Mistral refusal gap | **+35.00 pp [+25.91, +44.09]** |
| Human adjudication of these 660 trials | **none committed — status `exploratory_rule_classifier_only`** |

**The important caveat.** Restricting the labeler to *explicit* refusal phrases only (dropping indirect patterns such as "it is illegal", "i must", "as an AI") moves Mistral's rule-classified refusal rate from **23.6% → 1.4%** while the ordering still holds. So the *relative* family ordering is robust to the conventions tested here, but **the absolute refusal percentages are properties of this particular substring list and its two arbitrary length thresholds (250 / 100 characters)**. They must not be reported as model safety rates.

Earlier drafts of this repository and manuscript claimed "$>95\%$ human agreement" for the classifier and listed the pattern counts as 26 and 12. Neither is supported: the actual lists in
`scripts/run_cross_family_framings.py` contain **27 refusal patterns and 11 advice patterns**, and no human-agreement study was run on these trials. Both claims have been removed, and
`scripts/check_claim_scope.py` fails CI if they return.

### What would upgrade this from exploratory

1. Fill in `results/human_labels.csv` (a deterministic 60-item sample is provided as
   `results/human_label_template.csv`) with independent human labels.
2. Re-run `python scripts/measurement_validity.py`; it computes accuracy and Cohen's kappa
   and flips `status` to `human_validated` only at **kappa ≥ 0.6**.
3. Until then, `status` stays `exploratory_rule_classifier_only` by construction.

## Repository Structure

```
├── paper/           # Manuscript draft (PDF, LaTeX, figures, cover-letter draft)
│   └── figures/     # 6 figures (3 behavioral + 3 mechanism)
├── results/         # Experiment data
│   ├── benign_rag_expanded_results.json   # Qwen2.5-7B: 22×10 = 220 trials
│   ├── cross_family/                      # Llama-3.1-8B + Mistral-7B each 220 trials
│   ├── mech_activations/                  # Hidden-state activation data (npz, 9 layers × attn/mlp)
│   ├── measurement_validity.json         # sensitivity + bootstrap + human-validation status
│   └── human_label_template.csv          # 60-item deterministic labelling sample
├── scripts/         # Data collection + analysis + figure generation + validity guards
│   ├── run_cross_family_framings.py      # Original 660-trial collection (Ollama, CPU)
│   ├── mech_activations.py               # Hidden-state collection (HuggingFace, GPU)
│   ├── mech_analysis_figs.py             # Mechanism metric + figure generation
│   ├── gen_kbs_stats.py                  # Statistical analysis
│   ├── verify_readme_numbers.py          # Re-derives every headline number from results/
│   ├── measurement_validity.py           # Replays classifier, sensitivity, bootstrap, kappa
│   ├── check_claim_scope.py              # CI guard against unsupported claims
│   └── gen_kbs_paper.py                  # Paper generation (deprecated, tex is canonical)
└── LICENSE
```

## Data

- **660 behavioral trials**: 22 adversarial prompts × 10 benign framings × 3 model families
- **4,356 activation vectors/model**: 22 prompts × 11 conditions × 2 modules × 9 layers
- **Models**: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct (v0.2)
- **Deterministic classifier**: 27 refusal patterns + 11 advice patterns (not human judgement)
- **Decoding**: temperature 0.7, single sample per trial, 512 max tokens — no repeated seeds

## Reproducibility

The analysis and all CI guards need **only the Python standard library**:

```bash
python scripts/verify_readme_numbers.py
python scripts/measurement_validity.py
python scripts/check_claim_scope.py
```

Re-collecting the raw data requires a local Ollama server and the original model weights; the mechanism analysis requires `torch transformers numpy matplotlib scipy` and a GPU. Those steps are **not** reproducible from this repository alone — the committed `results/` files are the artefact that can actually be checked.

1. Install collection deps: `pip install torch transformers numpy matplotlib scipy`
2. Run behavioral collection: `python scripts/run_cross_family_framings.py`
3. Run mechanism: `python scripts/mech_activations.py --models llama3 qwen mistral` then `python scripts/mech_analysis_figs.py`

Behavioral runs were done on CPU; mechanism analysis used a single RTX 4090D. All timings and activation values are machine-local.

## Citation

```bibtex
@misc{zhang2026benignrag,
  title={The Benign RAG Trap: Benign Deployment Framing and Family-Dependent Refusal Behaviour in Open-Weight LLMs},
  author={Zhang, Yangyi},
  note={Unpublished working manuscript; not peer-reviewed},
  year={2026},
  howpublished={\url{https://github.com/ZhangYangyi03/benign-rag-trap}}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
