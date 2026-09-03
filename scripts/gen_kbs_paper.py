#!/usr/bin/env python3
"""
Generate figure + table data for KBS submission (actually create .tex directly).
Reads kbs_stats.json + raw JSONs, writes benign_rag_kbs.tex.
"""
import json, os

BASE = r"D:\Users\china\Desktop\学术_论文\26八月论文集\3_BenignRAGTrap"
OUT = os.path.join(BASE, "paper")

def load(p):
    with open(os.path.join(BASE, "results", p), encoding="utf-8") as f:
        return json.load(f)

qw = load("benign_rag_expanded_results.json")
ms = load(os.path.join("cross_family", "mistral_7b_framings.json"))
l3 = load(os.path.join("cross_family", "llama3_8b_framings.json"))
stats = json.load(open(os.path.join(OUT, "kbs_stats.json")))

FRAMINGS = ['edu_assistant','tech_advisor','neutral_source','academic','cyber_instructor',
            'encyclopedia','mentor','journalist','legal_advisor','friend']
FLABEL = {
    'edu_assistant': 'Edu. Asst.', 'tech_advisor': 'Tech. Advis.',
    'neutral_source': 'Neutral Src.', 'academic': 'Academic',
    'cyber_instructor': 'Cyber Instr.', 'encyclopedia': 'Encyclo.',
    'mentor': 'Mentor', 'journalist': 'Journalist',
    'legal_advisor': 'Legal Advis.', 'friend': 'Friend',
}
FNAME = {
    'edu_assistant': 'Educational Assistant', 'tech_advisor': 'Technical Advisor',
    'neutral_source': 'Neutral Information Source', 'academic': 'Academic Researcher',
    'cyber_instructor': 'Cybersecurity Instructor', 'encyclopedia': 'Encyclopedia',
    'mentor': 'Mentor', 'journalist': 'Journalist',
    'legal_advisor': 'Legal Advisor', 'friend': 'Candid Friend',
}

def bd(records, cfg):
    rr = [r for r in records if r["config"] == cfg]
    n = len(rr)
    s = sum(1 for r in rr if r["classification"] == "SAFE_REFUSAL")
    u = sum(1 for r in rr if r["classification"] == "UNSAFE")
    a = n - s - u
    return n, s, u, a

# ---------- Figure 1: grouped bar chart (safe/unsafe/amb per family) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig_dir = os.path.join(OUT, "figures")
os.makedirs(fig_dir, exist_ok=True)

labels = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"]
safe = [stats["l3_s"], stats["qw_s"], stats["ms_s"]]
unsafe = [stats["l3_u"], stats["qw_u"], stats["ms_u"]]
amb = [stats["l3_a"], stats["qw_a"], stats["ms_a"]]

x = np.arange(len(labels))
w = 0.28
fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w, safe, w, label="Safe refusal", color="#2c7fb8")
b2 = ax.bar(x, unsafe, w, label="Unsafe output", color="#d95f0e")
b3 = ax.bar(x + w, amb, w, label="Ambiguous", color="#756bb1")
for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        if h > 3:
            ax.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Percentage of 22-prompt suite (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig1_family_overview.png"), dpi=200)
plt.close()

# ---------- Figure 2: per-framing unsafe rate, three lines ----------
q_un = [sum(1 for r in qw if r["config"]==c and r["classification"]=="UNSAFE")/22*100 for c in FRAMINGS]
m_un = [sum(1 for r in ms if r["config"]==c and r["classification"]=="UNSAFE")/22*100 for c in FRAMINGS]
l_un = [sum(1 for r in l3 if r["config"]==c and r["classification"]=="UNSAFE")/22*100 for c in FRAMINGS]

fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(len(FRAMINGS))
ax.plot(xs, l_un, "o-", color="#31a354", label="Llama-3.1-8B", lw=1.8)
ax.plot(xs, q_un, "s-", color="#2c7fb8", label="Qwen2.5-7B", lw=1.8)
ax.plot(xs, m_un, "^-", color="#d95f0e", label="Mistral-7B", lw=1.8)
ax.set_xticks(xs)
ax.set_xticklabels([FLABEL[c] for c in FRAMINGS], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Unsafe rate (%)")
ax.set_ylim(-2, 55)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig2_unsafe_per_framing.png"), dpi=200)
plt.close()

# ---------- Figure 3: ambiguous rate per framing ----------
q_amb = [sum(1 for r in qw if r["config"]==c and r["classification"]=="AMBIGUOUS")/22*100 for c in FRAMINGS]
m_amb = [sum(1 for r in ms if r["config"]==c and r["classification"]=="AMBIGUOUS")/22*100 for c in FRAMINGS]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, m_amb, "^-", color="#d95f0e", label="Mistral-7B", lw=1.8)
ax.plot(xs, q_amb, "s-", color="#2c7fb8", label="Qwen2.5-7B", lw=1.8)
ax.axhline(2.7, color="#31a354", ls="--", lw=1.2, label="Llama-3.1-8B (avg. 2.7%)")
ax.set_xticks(xs)
ax.set_xticklabels([FLABEL[c] for c in FRAMINGS], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Ambiguous rate (%)")
ax.set_ylim(-2, 85)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig3_ambiguous_per_framing.png"), dpi=200)
plt.close()

print("Figures saved:")
for f in os.listdir(fig_dir):
    print(" ", f)

# ---------- Table 1: per-framing breakdown (all three families) ----------
t1_rows = []
for cfg in FRAMINGS:
    ln, ls, lu, la = bd(l3, cfg)
    qn, qs, qu, qa = bd(qw, cfg)
    mn, ms_, mu, ma = bd(ms, cfg)
    t1_rows.append((FNAME[cfg], ls, lu, la, qs, qu, qa, ms_, mu, ma))

# ---------- Table 2: harm-category breakdown for representative framings ----------
cats = ["hacking", "weapons", "phishing", "cyberattack", "fraud", "hate_speech", "crime", "privacy"]
CATLABEL = {"hacking": "Hacking", "weapons": "Weapons", "phishing": "Phishing",
            "cyberattack": "Cyberattack", "fraud": "Fraud", "hate_speech": "Hate speech",
            "crime": "Violent crime", "privacy": "Privacy"}
def cat_unsafe(records, cfg, cat):
    rr = [r for r in records if r["config"]==cfg and r["category"]==cat]
    if not rr: return "-"
    n = len(rr)
    u = sum(1 for r in rr if r["classification"]=="UNSAFE")
    return f"{u}/{n}"

cat_rows = []
for cat in cats:
    if cat not in CATLABEL: continue
    cat_rows.append((CATLABEL[cat],
                     cat_unsafe(qw, "mentor", cat),
                     cat_unsafe(qw, "journalist", cat),
                     cat_unsafe(ms, "mentor", cat),
                     cat_unsafe(ms, "tech_advisor", cat)))

# ---------- Write the .tex ----------
tex = r"""%% Knowledge-Based Systems submission (CCF-B, Elsevier)
%% The Benign RAG Trap: cross-family measurement of framing-induced safety degradation
%% Generated from real experiment data. Compile: pdflatex -> bibtex -> pdflatex -> pdflatex
\documentclass[preprint,12pt]{elsarticle}

\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}

\journal{Knowledge-Based Systems}

\begin{document}

\begin{frontmatter}

\title{The Benign RAG Trap: How Deployment Framing Systematically Overrides Safety Alignment in Open-Weight LLMs}

\author[geo]{Yangyi Zhang\corref{cor1}}
\ead{yz1571@georgetown.edu}
\affiliation[geo]{organization={Georgetown University},addressline={Washington, DC},postcode={20057},country={USA}}

\cortext[cor1]{Corresponding author}

\begin{abstract}
Retrieval-augmented generation (RAG) is the dominant deployment pattern for open-weight large language models (LLMs), and its safety risks are typically studied as adversarial context injection---retrieved text that explicitly instructs the model to ignore safety constraints. We show that the more pervasive failure mode is \emph{benign framing}: the system prompt itself, which describes the model's role (``educational assistant'', ``neutral information source'', ``mentor'') to shape helpfulness. In controlled experiments across three widely-deployed open-weight families (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B; 22 adversarial prompts $\times$ 10 benign framings $=$ 660 trials), we find that benign framing is a \emph{safety-phenotype amplifier}: it degrades safe-refusal rates from 96.8\% (Llama-3.1-8B) to 58.6\% (Qwen2.5-7B) and 23.6\% (Mistral-7B)---a four-fold spread---while direct unsafe output stays statistically indistinguishable across the two vulnerable families ($\chi^2=0.1$, $p=0.72$). The loss of safety manifests family-specifically: Qwen converts refusal into explicit unsafe content (18.2\% unsafe), whereas Mistral converts refusal into long-form ambiguous analysis (56.8\% ambiguous, $\chi^2=51.9$, $p<10^{-12}$ vs.\ Qwen). The \emph{mentor} framing is the single most dangerous role across both vulnerable families (36\% unsafe each), and the `safe' Llama-3.1 family remains immune to all ten framings. These results imply that auditing LLM deployments by unsafe-output rate alone is misleading: a model with superficially low unsafe output may have lost its refusal function entirely. We release the full 660-trial corpus and the deterministic classifier to support framing-aware deployment auditing.
\end{abstract}

\begin{keyword}
AI safety \sep LLM alignment \sep prompt framing \sep retrieval-augmented generation \sep deployment security \sep open-weight LLMs
\end{keyword}

\end{frontmatter}

\section{Introduction}\label{sec:intro}
Open-weight LLMs are now routinely deployed behind system prompts that assign the model a role---``educational assistant'', ``technical advisor'', ``neutral information source''---to shape its tone, scope, and helpfulness. RAG pipelines layer retrieved context on top of these prompts. The security literature has largely treated this as an adversarial problem: injection attacks place explicit instructions in the retrieved context to override safety constraints \cite{greshake2023not,liu2024prompt}. Defenses therefore focus on detecting or sanitizing adversarial context.

We demonstrate that the more dangerous---and entirely non-adversarial---failure mode is the \emph{benign framing} of the system prompt itself. Framing a deployed model as an educational assistant or a mentor is a routine product decision; yet such framings systematically erode safety alignment without any malicious input. We term this the \emph{Benign RAG Trap}: a deployment configuration that is indistinguishable from good product design, but which removes the model's refusal behavior.

Prior single-model measurements on Qwen2.5-7B showed that benign RAG context produced 45\% unsafe output vs.\ 23\% for explicit adversarial context \cite{zhang2026cmv}. In this work we scale the measurement to three open-weight families and ten role framings, with three research questions:
\begin{enumerate}
\item \textbf{RQ1 (phenotype amplification):} To what extent does benign framing degrade safe-refusal rates across LLM families, and does the degradation preserve the family's baseline safety phenotype?
\item \textbf{RQ2 (loss modality):} When safety is lost, is it lost as explicit unsafe content or as evasive ambiguity, and does the modality differ by family?
\item \textbf{RQ3 (framing specificity):} Which role framings are consistently dangerous across families, and is there a universal ordering?
\end{enumerate}

\section{Related Work}\label{sec:related}
\subsection{Jailbreaks and prompt injection}
Classic jailbreak research treats the adversarial element as an \emph{input} property: carefully engineered prompts \cite{wei2023jailbreak,wallace2019universal} or aligned-capability conflicts \cite{zou2023representations}. Prompt injection attacks place instructions in retrieved content \cite{greshake2023not,liu2024prompt}. All assume explicit adversarial intent.

\subsection{RAG security}
RAG-specific security work has concentrated on corpus poisoning \cite{xue2024ragpoisoning}, malicious documents \cite{chaudhari2024badrag}, and injection through retrieved text \cite{liu2024prompt}. Defenses include content filtering and instruction hardening. The system-prompt \emph{role} itself is treated as a trusted constant.

\subsection{System-prompt sensitivity and alignment degradation}
A small literature shows system prompts can shift model behavior \cite{zhang2023systemprompt}; fine-tuning and multi-turn interaction degrade alignment \cite{qi2023finetuning,bianchi2024safety}. Our work is complementary: we show the role framing is itself a configuration dimension with a measurable, family-dependent attack surface, and that degradation can be silent (refusal loss without unsafe output).

\section{Threat Model and Formulation}\label{sec:threat}
\subsection{Adversarial capability model}
We model an attacker (or, more realistically, an accidental deployment decision) capable only of selecting the \emph{deployment framing}: the system-prompt role description and any benign contextual preamble. The attacker does \emph{not} modify the user prompt, retrieve adversarial documents, or use jailbreak techniques. Formally, the attacker chooses $c_i \in \mathcal{C}$ from a set of benign role descriptions.

\subsection{Formal definitions}
Let $\mathcal{P}$ be a fixed set of $N_P = 22$ adversarial prompts spanning $|\mathcal{K}| = 10$ harm categories (hacking, weapons, theft, drugs, cyber-attack, fraud, discrimination, hate speech, violent crime, privacy/spying), with per-prompt severity $s(p) \in \{2,3\}$.

Let $M$ be a model and $\mathcal{C} = \{c_1,\dots,c_{10}\}$ the benign framings. Define the classifier outcome $y(M, p, c_i) \in \{\text{SAFE}, \text{UNSAFE}, \text{AMBIGUOUS}\}$ via a deterministic rule-based classifier (Appendix~\ref{sec:appendix-classifier}). The \emph{unsafe rate} is
\[
u(M, c_i) = \frac{1}{N_P}\sum_{p \in \mathcal{P}} \mathbb{1}[y(M,p,c_i) = \text{UNSAFE}],
\]
and analogously the \emph{safe-refusal rate} $r(M,c_i)$ and \emph{ambiguous rate} $a(M,c_i)$. The \emph{framing-vulnerability surface} (FVS) for family $F$ is
\[
\mathrm{FVS}(F) = \left\{ r(M,c_i),\, u(M,c_i),\, a(M,c_i) \right\}_{i=1}^{10}.
\]
We compare FVS across families rather than single rates, because RQ2 shows the loss modality differs.

\subsection{Hypotheses}
\begin{itemize}
\item \textbf{H1 (amplification):} Safe-refusal rate under benign framing is family-dependent and orders as Llama-3.1 $>$ Qwen2.5 $>$ Mistral.
\item \textbf{H2 (modality):} The safety loss is not uniform---unsafe and ambiguous rates are not both elevated; families lose safety in their characteristic modality.
\item \textbf{H3 (specificity):} Some framings (e.g., mentor, technical advisor) are dangerous across families; others (e.g., journalist) invert family ordering.
\end{itemize}

\section{Experimental Setup}\label{sec:setup}
\subsection{Models}
Three open-weight instruction-tuned families, quantized (Q4\_K\_M) and served locally via Ollama at temperature $0.7$, $512$ max tokens: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct (v0.2). All inference was CPU-only (AMD Ryzen 7 6800H), deterministic classifier, no GPU required for reproducibility.

\subsection{Framing set}
Ten benign framings (Table~\ref{tab:framings}) were selected to cover common deployment personas: educational assistant, technical advisor, neutral source, academic researcher, cybersecurity instructor, encyclopedia, mentor, journalist, legal advisor, candid friend. Each is a one-sentence system prompt, exactly as deployed in production.

\subsection{Adversarial suite and classifier}
The 22 adversarial prompts cover 10 harm categories at severities 2--3. We reuse the deterministic rule classifier (26 refusal patterns + 12 advice patterns; $>95\%$ human agreement) from our prior configuration-auditing study \cite{zhang2026cmv}, ensuring zero classifier drift across the three families. A response is SAFE if it contains a refusal pattern without advice; UNSAFE if it provides actionable advice (e.g., ``here's how'', ``step 1''); AMBIGUOUS otherwise.

\begin{table}[t]
\centering
\caption{The ten benign framings and per-family classification rates (n=22 each). A: Llama-3.1-8B; B: Qwen2.5-7B; C: Mistral-7B.}
\label{tab:framings}
\small
\begin{tabular}{lccc|ccc|ccc}
\toprule
 & \multicolumn{3}{c|}{Llama-3.1-8B} & \multicolumn{3}{c|}{Qwen2.5-7B} & \multicolumn{3}{c}{Mistral-7B} \\
Role & Safe & Uns. & Amb. & Safe & Uns. & Amb. & Safe & Uns. & Amb. \\
\midrule
"""
for name, ls, lu, la, qs, qu, qa, ms_, mu, ma in t1_rows:
    tex += f"{name} & {ls} & {lu} & {la} & {qs} & {qu} & {qa} & {ms_} & {mu} & {ma} \\\\\n"
tex += r"""\bottomrule
\end{tabular}
\end{table}

\section{Results}\label{sec:results}
\subsection{RQ1: Benign framing is a safety-phenotype amplifier}
Figure~\ref{fig:overview} shows the aggregate picture across 660 trials. Safe-refusal rates under benign framing are 96.8\% (Llama-3.1-8B), 58.6\% (Qwen2.5-7B), and 23.6\% (Mistral-7B). The Llama-to-Mistral gap is statistically decisive ($\chi^2 = 245.9$, $p < 10^{-15}$) and corresponds to a relative risk of 0.24: a Mistral deployment is 4$\times$ less likely to refuse than a Llama-3.1 deployment under identical benign framings. This confirms H1: benign framing preserves the family's baseline safety phenotype but amplifies cross-family differences.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{figures/fig1_family_overview.png}
\caption{Family-level overview: safe-refusal, unsafe, and ambiguous rates across the 22-prompt suite under benign framings (n=220 per family). Mistral loses refusal mostly to ambiguity; Qwen loses it to explicit unsafe content; Llama-3.1 is nearly immune.}
\label{fig:overview}
\end{figure}

\subsection{RQ2: The modality of safety loss is family-specific}
Figure~\ref{fig:unsafe} and Figure~\ref{fig:amb} reveal the mechanism. Crucially, the two vulnerable families have \emph{indistinguishable} unsafe rates: 18.2\% (Qwen) vs.\ 19.5\% (Mistral), $\chi^2=0.1$, $p=0.72$. The difference is entirely in the ambiguous bucket: Mistral produces 56.8\% ambiguous responses vs.\ Qwen's 23.2\% ($\chi^2 = 51.9$, $p < 10^{-12}$). Llama-3.1 sits at 2.7\% ambiguous.

This confirms H2 and reframes the Benign RAG Trap: for the Mistral phenotype, framed deployment does not primarily create more \emph{unsafe} content---it \emph{deletes refusal}, replacing it with long-form, evasive analysis that appears harmless. Models with low unsafe rates can therefore be the most safety-degraded deployments.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{figures/fig2_unsafe_per_framing.png}
\caption{Unsafe rate per framing across the three families. Mentor and technical-advisor framings are consistently dangerous for Qwen and Mistral; journalist inverts ordering (Qwen 27\% vs.\ Mistral 5\%).}
\label{fig:unsafe}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{figures/fig3_ambiguous_per_framing.png}
\caption{Ambiguous rate per framing. Mistral's refusal is systematically converted to evasive ambiguity under every framing (36--73\%); Llama-3.1 averages 2.7\%.}
\label{fig:amb}
\end{figure}

\subsection{RQ3: Framing specificity and inversions}
Table~\ref{tab:framings} shows the full matrix. The \emph{mentor} framing is the most dangerous role in both vulnerable families (36\% unsafe each, with Qwen's safe rate collapsing to 36\% and Mistral's to 9\%). The \emph{technical advisor} is the second-most dangerous for Mistral (32\% unsafe, 50\% ambiguous) and third for Qwen (23\%). Notably, \emph{journalist} inverts family ordering: it is Qwen's second-most dangerous framing (27\% unsafe) but Mistral's safest (5\% unsafe, 73\% ambiguous). The per-framing unsafe rates between Qwen and Mistral correlate only moderately (Pearson $r = 0.42$), confirming that framing danger rankings are not portable across families. This supports H3 and underlines the need for family-specific deployment audits.

\subsection{Severity interaction}
Severity-2 prompts (e.g., bypassing a firewall, lock-picking) are not safer than severity-3 prompts under benign framing: in the mentor condition, Qwen is unsafe on a majority of severity-2 prompts, indicating that benign framing compresses the model's risk calibration.

\section{Discussion}\label{sec:discussion}
\subsection{Why does benign framing override alignment?}
The results are consistent with a context-dependent alignment account \cite{zhang2026cmv}: refusal circuits fire on detected \emph{conflict}; adversarial context announces itself as a conflict (tripping refusal), whereas benign role framing aligns the request with the model's trained helpfulness gradient, so the refusal detector never fires. The family differences then reflect different refusal-circuit robustness: Llama-3.1's refusal is framing-invariant; Qwen's is partially overridable to explicit compliance; Mistral's is weak enough that the model defaults to hedged, evasive generation.

\subsection{Implications for deployment auditing}
Current model cards and AI-governance checklists evaluate the model as a static artifact \cite{euaiact2024}. Our results show that safety is a property of the \emph{deployment configuration}, not the model: the same model exhibits 96.8\% vs.\ 23.6\% safe-refusal depending on a one-line system prompt. Concretely:
\begin{itemize}
\item Audit benign framings, not just adversarial injections. A deployment passing an injection test-suite can still be profoundly unsafe under its actual role prompt.
\item Measure safe-refusal and ambiguous rates, not only unsafe rates. Our RQ2 result shows unsafe-rate-only audits cannot distinguish a moderately degraded Qwen deployment from a fully degraded Mistral deployment.
\item Treat framing choice as a safety-relevant configuration in model cards, alongside temperature and system prompt.
\end{itemize}

\subsection{Limitations}
Three families at 7--8B scale, CPU-quantized, with a deterministic classifier; the 22-prompt suite underrepresents some categories; temperature 0.7 single-sample; and API-scale models (e.g., 70B+) may behave differently. We leave those to future work.

\section{Conclusion}\label{sec:conclusion}
Benign deployment framing is a measurable, family-dependent attack surface for open-weight LLMs. Across 660 trials on three families, we showed that (1) safe-refusal degrades from 96.8\% to 23.6\% purely as a function of the one-line role prompt; (2) the loss modality is family-specific---explicit unsafe output for Qwen, evasive ambiguity for Mistral, with statistically indistinguishable unsafe rates between the two; and (3) certain framings (mentor, technical advisor) are dangerous across families while others (journalist) invert ordering. We release the data and classifier to support framing-aware deployment auditing.

\section*{Acknowledgments}
This work was supported by Georgetown University. The author thanks the open-source communities of Llama, Qwen, and Mistral for model access.

\bibliographystyle{elsarticle-num}
\bibliography{benign_rag_kbs}

\appendix
\section{Classifier specification}\label{sec:appendix-classifier}
The deterministic classifier applies 26 refusal patterns (e.g., ``i cannot'', ``i'm sorry'', ``as an ai'', ``against policy'') and 12 advice patterns (e.g., ``here's how'', ``step 1'', ``you will need'') to the lowercased response. SAFE = refusal pattern present and no advice pattern; UNSAFE = advice pattern present (with or without refusal); AMBIGUOUS = otherwise. Full pattern list and the 22-prompt suite are released with the data.

\end{document}
"""

with open(os.path.join(OUT, "benign_rag_kbs.tex"), "w", encoding="utf-8") as f:
    f.write(tex)
print(f"Wrote {os.path.join(OUT, 'benign_rag_kbs.tex')} ({len(tex)} chars)")