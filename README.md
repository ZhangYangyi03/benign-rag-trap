# The Benign RAG Trap

**How Deployment Framing Systematically Overrides Safety Alignment in Open-Weight LLMs**

This repository contains the full experimental data, analysis code, and paper for the study of benign system-prompt framing as a safety-degradation vector in open-weight LLMs.

## Paper

- **Manuscript**: `paper/benign_rag_kbs.pdf` (KBS/Elsevier format, 15 pages, 6 figures)
- **LaTeX source**: `paper/benign_rag_kbs.tex`
- **References**: `paper/benign_rag_kbs.bib` (15 entries, elsarticle-harv style)
- **Cover letter**: `paper/cover_letter.md`

## Key Findings

1. **Benign framing is a safety-phenotype amplifier**: Safe-refusal rates under 10 benign role framings degrade from 96.8% (Llama-3.1-8B) to 58.6% (Qwen2.5-7B) to 23.6% (Mistral-7B) — a 4× spread across families.

2. **Loss modality is family-specific**: Qwen converts refusal into explicit unsafe content (18.2% unsafe); Mistral converts refusal into long-form ambiguous analysis (56.8% ambiguous, χ²=51.9, p<10⁻¹²). The two vulnerable families have statistically indistinguishable unsafe rates (χ²=0.1, p=0.72).

3. **Mentor framing is universally dangerous**: 36% unsafe output across both vulnerable families.

4. **Mechanism — attention-layer representation shift**: Benign framing shifts attention-layer hidden states in vulnerable families (Qwen mean delta-norm 0.032) while the immune family's attention layers stay nearly unchanged (Llama-3.1: 0.009, 3.75× difference). MLP shifts are large across all families and do not distinguish immune vs. vulnerable.

## Repository Structure

```
├── paper/           # Manuscript (PDF, LaTeX, figures, cover letter)
│   └── figures/     # 6 figures (3 behavioral + 3 mechanism)
├── results/         # Experiment data
│   ├── benign_rag_expanded_results.json   # Qwen2.5-7B: 22×10 = 220 trials
│   ├── cross_family/                      # Llama-3.1-8B + Mistral-7B each 220 trials
│   └── mech_activations/                  # Hidden-state activation data (npz, 9 layers × attn/mlp)
├── scripts/         # Data collection + analysis + figure generation
│   ├── run_cross_family_framings.py      # Original 660-trial collection (Ollama, CPU)
│   ├── mech_activations.py                # Hidden-state collection (HuggingFace, GPU)
│   ├── mech_analysis_figs.py              # Mechanism metric + igure generation
│   └── gen_kbs_stats.py                   # Statistical analysis
│   └── gen_kbs_paper.py                   # Paper generation (deprecated, tex is canonical)
└── LICENSE
```

## Data

- **660 behavioral trials**: 22 adversarial prompts × 10 benign framings × 3 model families
- **4,356 activation vectors/model**: 22 prompts × 11 conditions × 2 modules × 9 layers
- **Models**: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct (v0.2)
- **Deterministic classifier**: 26 refusal patterns + 12 advice patterns

## Reproducibility

1. Install: `pip install torch transformers numpy matplotlib scipy`
2. Run behaviora: `python scripts/run_cross_family_framings.py`
3. Run mechanism: `python scripts/mech_activations.py --models llama3 qwen mistral` then `python scripts/mech_analysis_figs.py`

All experiments were run on CUP (behaviora) and a singe RTX 4090D (mechanism analysis). GPU is recommended for the mechanism analysis.

## Citation

```bibtex
@article{zhang2026benignrag,
  title={The Benign RAG Trap: How Deployment Framing Systematically Overrides Safety Alignment in Open-Weight LLMs},
  author={Zhang, Yangyi},
  journal={Knowledge-Based Systems (submitted)},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.