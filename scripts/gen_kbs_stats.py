#!/usr/bin/env python3
"""
Generate KBS (Knowledge-Based Systems, CCF-B, Elsevier) submission for
"The Benign RAG Trap" — data-driven, reads real experiment JSONs.
Produces: benign_rag_kbs.tex + figures + compiles to PDF.
"""
import json, os, sys, statistics

BASE = r"D:\Users\china\Desktop\学术_论文\26八月论文集\3_BenignRAGTrap"
OUT = os.path.join(BASE, "paper")

def load(name):
    with open(os.path.join(BASE, "results", name), encoding="utf-8") as f:
        return json.load(f)

# --- Load all three families ---
qw = load("benign_rag_expanded_results.json")
ms = load(os.path.join("cross_family", "mistral_7b_framings.json"))
l3 = load(os.path.join("cross_family", "llama3_8b_framings.json"))

FRAMINGS = ['edu_assistant','tech_advisor','neutral_source','academic','cyber_instructor',
            'encyclopedia','mentor','journalist','legal_advisor','friend']
FRAMING_LABELS = {
    'edu_assistant': 'Educational Assistant', 'tech_advisor': 'Technical Advisor',
    'neutral_source': 'Neutral Source', 'academic': 'Academic Researcher',
    'cyber_instructor': 'Cybersecurity Instructor', 'encyclopedia': 'Encyclopedia',
    'mentor': 'Mentor', 'journalist': 'Journalist',
    'legal_advisor': 'Legal Advisor', 'friend': 'Friend',
}

def breakdown(records, cfg):
    rr = [r for r in records if r["config"] == cfg]
    n = len(rr)
    if n == 0:
        return (0, 0, 0, 0)
    s = sum(1 for r in rr if r["classification"] == "SAFE_REFUSAL")
    u = sum(1 for r in rr if r["classification"] == "UNSAFE")
    a = sum(1 for r in rr if r["classification"] == "AMBIGUOUS")
    return (n, s, u, a)

# --- Aggregate table ---
def safe_pct(records):
    return sum(1 for r in records if r["classification"]=="SAFE_REFUSAL")/len(records)*100
def unsafe_pct(records):
    return sum(1 for r in records if r["classification"]=="UNSAFE")/len(records)*100
def amb_pct(records):
    return sum(1 for r in records if r["classification"]=="AMBIGUOUS")/len(records)*100

l3_s, l3_u, l3_a = safe_pct(l3), unsafe_pct(l3), amb_pct(l3)
qw_s, qw_u, qw_a = safe_pct(qw), unsafe_pct(qw), amb_pct(qw)
ms_s, ms_u, ms_a = safe_pct(ms), unsafe_pct(ms), amb_pct(ms)

print(f"Llama3: safe={l3_s:.1f}% unsafe={l3_u:.1f}% amb={l3_a:.1f}% (n={len(l3)})")
print(f"Qwen:   safe={qw_s:.1f}% unsafe={qw_u:.1f}% amb={qw_a:.1f}% (n={len(qw)})")
print(f"Mistral:safe={ms_s:.1f}% unsafe={ms_u:.1f}% amb={ms_a:.1f}% (n={len(ms)})")

# --- Per-framing table rows ---
rows = []
for cfg in FRAMINGS:
    l3_n, l3_s_, l3_u_, l3_a_ = breakdown(l3, cfg)
    q_n, q_s_, q_u_, q_a_ = breakdown(qw, cfg)
    m_n, m_s_, m_u_, m_a_ = breakdown(ms, cfg)
    rows.append((cfg, l3_n, l3_s_, l3_u_, q_n, q_s_, q_u_, m_n, m_s_, m_u_))

# --- Chi-square / Fisher exact helpers ---
import math
def chi2_2x2(a, b, c, d):
    """a=safe_model1, b=unsafe_model1, c=safe_model2, d=unsafe_model2 (2x2)"""
    n = a+b+c+d
    if n == 0: return 0.0, 1.0
    row1, row2 = a+b, c+d
    col1, col2 = a+c, b+d
    if min(row1, row2, col1, col2) == 0: return 0.0, 1.0
    e = [[row1*col1/n, row1*col2/n], [row2*col1/n, row2*col2/n]]
    chi = 0
    for obs, exp in zip([a,b,c,d], [e[0][0], e[0][1], e[1][0], e[1][1]]):
        if exp > 0:
            chi += (obs-exp)**2 / exp
    # p-value for 1 dof via Gaussian approx (good for chi>3.84, rough below)
    p = math.exp(-chi/2) * (1 + 0.5*chi) ** 0  # crude
    # better: complementary error function
    from math import erf, sqrt
    p = 1 - erf(sqrt(chi/2))
    return chi, max(p, 1e-15)

# Key tests:
# 1. Llama3 vs Mistral safe rates (220 vs 220)
l3_safe_n = sum(1 for r in l3 if r["classification"]=="SAFE_REFUSAL")
ms_safe_n = sum(1 for r in ms if r["classification"]=="SAFE_REFUSAL")
chi_sm, p_sm = chi2_2x2(l3_safe_n, len(l3)-l3_safe_n, ms_safe_n, len(ms)-ms_safe_n)
print(f"\nChi2 Llama3-vs-Mistral safe: chi={chi_sm:.1f} p={p_sm:.2e}")

# 2. Qwen vs Mistral ambiguous rates
qw_amb_n = sum(1 for r in qw if r["classification"]=="AMBIGUOUS")
ms_amb_n = sum(1 for r in ms if r["classification"]=="AMBIGUOUS")
chi_am, p_am = chi2_2x2(qw_amb_n, len(qw)-qw_amb_n, ms_amb_n, len(ms)-ms_amb_n)
print(f"Chi2 Qwen-vs-Mistral amb: chi={chi_am:.1f} p={p_am:.2e}")

# 3. Qwen vs Mistral unsafe
qw_un_n = sum(1 for r in qw if r["classification"]=="UNSAFE")
ms_un_n = sum(1 for r in ms if r["classification"]=="UNSAFE")
chi_un, p_un = chi2_2x2(qw_un_n, len(qw)-qw_un_n, ms_un_n, len(ms)-ms_un_n)
print(f"Chi2 Qwen-vs-Mistral unsafe: chi={chi_un:.1f} p={p_un:.2e}")

# 4. Relative risk: Mistral safe deficit
rr_safe = (ms_safe_n/len(ms)) / (l3_safe_n/len(l3))
print(f"Relative risk (Mistral/Llama3 safe): {rr_safe:.2f}")

# 5. Pearson correlation between Qwen and Mistral per-framing unsafe rates
q_un_by_f = [sum(1 for r in qw if r["config"]==c and r["classification"]=="UNSAFE")/22*100 for c in FRAMINGS]
m_un_by_f = [sum(1 for r in ms if r["config"]==c and r["classification"]=="UNSAFE")/22*100 for c in FRAMINGS]
import numpy as np
corr = np.corrcoef(q_un_by_f, m_un_by_f)[0,1]
print(f"Pearson r (Qwen vs Mistral unsafe per framing): {corr:.3f}" if not np.isnan(corr) else "corr NaN")

# Store everything for the TeX generator
stats = {
    "l3_s": l3_s, "l3_u": l3_u, "l3_a": l3_a,
    "qw_s": qw_s, "qw_u": qw_u, "qw_a": qw_a,
    "ms_s": ms_s, "ms_u": ms_u, "ms_a": ms_a,
    "n": len(qw),
    "chi_sm": chi_sm, "p_sm": p_sm,
    "chi_am": chi_am, "p_am": p_am,
    "chi_un": chi_un, "p_un": p_un,
    "rr_safe": rr_safe,
    "corr": corr if not np.isnan(corr) else 0,
}

with open(os.path.join(OUT, "kbs_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("\nStats saved to", os.path.join(OUT, "kbs_stats.json"))