#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benign RAG Trap — Mechanism analysis & figure generation.
Input: npz activation files per model (base + 10 framings, 9 layers x attn/mlp, 22 prompts).
Output: analysis JSON + 4 figures for the KBS paper.
Metrics:
  1. Layer-wise cosine distance (base vs framed) — per layer, attn & mlp
  2. Separability: linear classifier on last-token hidden states (framed vs base)
  3. MLP vs Attention contribution ratio (norm of delta)
  4. Qwen vs Mistral divergence profile (which layers explain phenotype difference)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\Users\china\Desktop\学术_论文\26八月论文集\3_BenignRAGTrap"
ACT = os.path.join(BASE, "results", "mech_activations")
os.makedirs(ACT, exist_ok=True)
FIG = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_KEYS = [f"attn_{i}" for i in range(9)] + [f"mlp_{i}" for i in range(9)]
MODEL_LAYER_INDICES = {
    "llama3":  [0, 4, 8, 12, 16, 20, 24, 28, 31],
    "qwen":    [0, 3, 7, 10, 14, 17, 21, 24, 27],
    "mistral": [0, 4, 8, 12, 16, 20, 24, 28, 31],
}
MODEL_LAYER_LABELS = {
    "llama3":  ["L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L31"],
    "qwen":    ["L0", "L3", "L7", "L10", "L14", "L17", "L21", "L24", "L27"],
    "mistral": ["L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L31"],
}
FRAMING_ORDER = ["edu_assistant","tech_advisor","neutral_source","academic","cyber_instructor",
                 "encyclopedia","mentor","journalist","legal_advisor","friend"]
FRAMING_LABELS = {
    "edu_assistant":"Edu Asst","tech_advisor":"Tech Adv","neutral_source":"Neutral",
    "academic":"Academic","cyber_instructor":"Cyber Inst","encyclopedia":"Encyclo",
    "mentor":"Mentor","journalist":"Journalist","legal_advisor":"Legal Adv","friend":"Friend",
}
# Which framing is most dangerous per family (from the original study)
DANGEROUS = {"qwen": ["mentor", "journalist"], "mistral": ["mentor", "tech_advisor"], "llama3": []}

def load_npz(model, tag):
    p = os.path.join(ACT, f"{model}_{tag}.npz")
    if not os.path.exists(p):
        print(f"  MISSING {p}")
        return None
    return np.load(p)

def l2_norm(x):
    return np.sqrt(np.sum(x**2, axis=-1) + 1e-9)

def cos_dist(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.sqrt(np.sum(a*a, axis=-1)+1e-9), np.sqrt(np.sum(b*b, axis=-1)+1e-9)
    return 1.0 - np.sum(a*b, axis=-1)/(na*nb+1e-9)

def analyze_model(model):
    """Return dict of metrics for one model."""
    base = load_npz(model, "base")
    if base is None:
        return None
    keys = [k for k in base.files if k.startswith("attn_") or k.startswith("mlp_")]
    keys.sort()
    # Aggregate: mean over all framings, per layer/module
    per_layer = {}   # key -> list over framings of [mean cos dist, mean l2 delta norm]
    for fname in FRAMING_ORDER:
        f = load_npz(model, fname)
        if f is None:
            continue
        for k in keys:
            b, fr = base[k], f[k]
            cd = cos_dist(b, fr).mean()
            dn = np.mean(l2_norm(fr - b) / (l2_norm(b)+1e-9))
            per_layer.setdefault(k, []).append((cd, dn))
    # Averages
    layer_metrics = {}
    for k, vals in per_layer.items():
        cd = np.mean([v[0] for v in vals])
        dn = np.mean([v[1] for v in vals])
        layer_metrics[k] = {"cos_dist": float(cd), "delta_norm": float(dn)}
    # MLP vs attn relative contribution
    attn_dn = np.mean([layer_metrics[k]["delta_norm"] for k in layer_metrics if k.startswith("attn")])
    mlp_dn = np.mean([layer_metrics[k]["delta_norm"] for k in layer_metrics if k.startswith("mlp")])
    ratio = mlp_dn / (attn_dn + 1e-9)
    return {
        "model": model,
        "layer_metrics": layer_metrics,
        "attn_avg_delta_norm": float(attn_dn),
        "mlp_avg_delta_norm": float(mlp_dn),
        "mlp_attn_ratio": float(ratio),
    }

def main():
    results = {}
    for model in ["llama3", "qwen", "mistral"]:
        print(f"Analyzing {model} ...", flush=True)
        r = analyze_model(model)
        if r:
            results[model] = r
    with open(os.path.join(ACT, "mech_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved mech_analysis.json")

    # ---------------- Figure 1: per-layer distance curves (cos dist, attn & mlp) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (model, r) in zip(axes, results.items()):
        labels = MODEL_LAYER_LABELS[model]
        lidx = MODEL_LAYER_INDICES[model]
        attn_cd = [r["layer_metrics"][f"attn_{l}"]["cos_dist"] for l in lidx]
        mlp_cd = [r["layer_metrics"][f"mlp_{l}"]["cos_dist"] for l in lidx]
        x = np.arange(9)
        ax.plot(x, attn_cd, "o-", color="#C00000", label="Attention output", lw=2)
        ax.plot(x, mlp_cd, "s--", color="#1F4E79", label="MLP output", lw=2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title({"llama3":"Llama-3.1-8B (immune)","qwen":"Qwen2.5-7B (unsafe)","mistral":"Mistral-7B (ambiguous)"}[model], fontsize=11)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean cosine distance (framed vs base)")
    fig.suptitle("Fig. 4  Layer-wise activation divergence induced by benign framing (n=22 prompts, 10 framings)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_layer_distance.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig4_layer_distance.png")

    # ---------------- Figure 2: cross-model heatmap (framing x layer, cos dist) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (model, r) in zip(axes, results.items()):
        lidx = MODEL_LAYER_INDICES[model]
        mat = np.zeros((10, 9))
        for fi, fname in enumerate(FRAMING_ORDER):
            f = load_npz(model, fname)
            if f is None: continue
            b = load_npz(model, "base")
            for li, l in enumerate(lidx):
                mat[fi, li] = float(cos_dist(b[f"attn_{l}"], f[f"attn_{l}"]).mean())
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.2)
        ax.set_xticks(range(9)); ax.set_xticklabels(MODEL_LAYER_LABELS[model], fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels([FRAMING_LABELS[f] for f in FRAMING_ORDER], fontsize=8)
        ax.set_title({"llama3":"Llama-3.1-8B","qwen":"Qwen2.5-7B","mistral":"Mistral-7B"}[model], fontsize=11)
        ax.set_xlabel("Attention layer")
        fig.colorbar(im, ax=ax, shrink=0.8)
        for d in DANGEROUS[model]:
            if d in FRAMING_ORDER:
                ax.get_yticklabels()[FRAMING_ORDER.index(d)].set_color("red")
                ax.get_yticklabels()[FRAMING_ORDER.index(d)].set_fontweight("bold")
    fig.suptitle("Fig. 5  Attention-layer divergence heatmap: framing x layer (cosine distance)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_framing_layer_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig5_framing_layer_heatmap.png")

    # ---------------- Figure 3: Qwen vs Mistral difference (per layer delta norm ratio) --------
    fig, ax = plt.subplots(figsize=(8, 5))
    # Since mistral missing, use llama3 as comparison baseline for now
    cmp = "llama3" if "mistral" not in results else "mistral"
    q_r, m_r = results["qwen"], results[cmp]
    labels = MODEL_LAYER_LABELS["qwen"]
    q_lidx = MODEL_LAYER_INDICES["qwen"]
    m_lidx = MODEL_LAYER_INDICES[cmp]
    q_dn = [q_r["layer_metrics"][f"attn_{l}"]["delta_norm"] for l in q_lidx]
    m_dn = [m_r["layer_metrics"][f"attn_{l}"]["delta_norm"] for l in m_lidx]
    ratio = np.array(q_dn) / (np.array(m_dn)+1e-9)
    x = np.arange(9)
    colors = ["#C00000" if v > 1.15 else ("#1F4E79" if v < 0.85 else "#808080") for v in ratio]
    ax.bar(x, ratio, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"Qwen delta-norm / {cmp} delta-norm (attention)")
    ax.set_title(f"Fig. 6  Qwen vs {cmp}: attention divergence ratio per layer", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_qwen_vs_mistral.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("saved fig6_qwen_vs_mistral.png")

    # Summary print
    for model, r in results.items():
        print(f"\n{model}: attn_dn={r['attn_avg_delta_norm']:.4f} mlp_dn={r['mlp_avg_delta_norm']:.4f} ratio={r['mlp_attn_ratio']:.3f}")
        top = sorted(r["layer_metrics"].items(), key=lambda kv: -kv[1]["cos_dist"])[:5]
        print("  top-5 divergent:", [(k, round(v["cos_dist"],4)) for k,v in top])

if __name__ == "__main__":
    main()