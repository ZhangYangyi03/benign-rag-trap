Dear Editor,

I am pleased to submit our manuscript entitled "The Benign RAG Trap: How Deployment Framing Systematically Overrides Safety Alignment in Open-Weight LLMs" for consideration for publication in Knowledge-Based Systems.

**Why KBS?**

This paper addresses a fundamental gap in the safety evaluation of knowledge-based AI systems. The benign RAG trap — where a system prompt that assigns the model a helpful role (e.g., "educational assistant", "mentor") inadvertently erodes safety alignment — is a direct threat to the reliability of knowledge-based LLM deployments. KBS's scope encompasses "knowledge-based systems, knowledge engineering, and intelligent decision support," which directly aligns with our contribution: we show that the knowledge delivery system (the system prompt and RAG pipeline) is not a neutral channel but an active safety-relevant configuration dimension.

**What this paper contributes:**

1. A controlled measurement of benign framing effects across three open-weight LLM families (660 trials), showing that safe-refusal degrades from 96.8% to 23.6% purely as a function of a one-line role prompt.

2. A novel mechanistic analysis using transformer hidden-state hooks, demonstrating that attention-layer representation shift is the mechanism of framing-induced safety degradation — the immune family's attention layers remain framing-invariant, while vulnerable families show significant shifts in the same layers identified as safety-critical detection circuits.

3. All data, classifier, and analysis scripts are released for reproducibility, consistent with KBS's emphasis on reproducible research.

**Novelty and significance:**

While prior work has studied adversarial prompt injection, no prior study has systematically measured the effect of benign role framing across multiple model families, nor has any work provided a mechanistic explanation via activation analysis. Our finding that the \emph{mentor} framing consistently causes 36% unsafe output across both vulnerable families has immediate practical implications for deployment safety policies.

The manuscript has not been published previously and is not under consideration elsewhere. All authors have approved the submission.

Thank you for your consideration.

Sincerely,
Yangyi Zhang
Georgetown University