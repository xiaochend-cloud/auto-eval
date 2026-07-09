# Repr-Probing for PID under Extreme Class Imbalance — Experiment Design (v0)

*Working design. Anchor = base-rate fallacy. Lens = representation-space probing. Not a neat trick; a multi-part mechanistic study.*

---

## 0. One-sentence thesis

> Under production base rates (~1000:1 benign:malicious), we **open the black box** — the detector's embedding space and decision boundary — to explain *how* data-side and loss-side interventions reshape the boundary, and *why* they help or hurt. The novelty is the **mechanistic explanation** (repr + boundary), not the interventions themselves.

## 1. Why this is defensible novelty (prior-work map)

| Prior work | What they do | What they DON'T do (our lane) |
|---|---|---|
| **When Benchmarks Lie** (Fomin, arXiv 2602.14161, Feb 2026) | OOD eval via **LODO**; SAE **shortcut-feature** analysis; evaluation-only, no method | No class-imbalance modeling, no asymmetric/focal loss, **no label-noise / near-boundary analysis**, no method |
| **Ayub & Majumdar** (CAMLIS 2024) | *Visualize* embeddings (PCA/t-SNE/UMAP), note benign/malicious overlap | Don't **diagnose** the overlap (label noise? boundary?); no imbalance; no intervention |
| **PromptShield** (CODASPY 2025) | Eval at fixed low FPR | Treat low-FPR as threshold calibration, not a **base-rate/imbalance modeling** problem |
| **InjecGuard / PIGuard** (ACL 2025) | Over-defense (false positives on trigger words) | The FP *side* only; not imbalance-as-modeling |

**Our unoccupied intersection:** imbalance-as-a-modeling-problem ⊕ label-noise/decision-boundary diagnosis ⊕ representation-space explanation. OOD is a *supporting* robustness axis (we adopt LODO-style cross-dataset eval and cite Fomin), **not** the headline.

## 2. The four probes (each maps to a paper claim)

### Probe 1 — Representation under training-ratio reshape
- **Question:** as train ratio goes balanced → 1000:1, how does the decision boundary / margin move?
- **Hypothesis H1:** the benign majority pushes the boundary *into* the malicious region; the margin distribution shifts so malicious points get smaller/negative margins → recall collapses. Visible as: malicious cluster drifting to the wrong side of the boundary as ratio grows.
- **Measures:** margin (signed distance to boundary = logit) distribution per class across ratios; recall@fixed-FPR; boundary angle/shift.

### Probe 2 — Representation of ambiguous (removed-benign) data
- **Question:** where do the ambiguous-benign examples (whose removal improved performance) sit?
- **Hypothesis H2:** they sit **near the boundary** (|margin|≈0, prob≈0.5) with **mixed-label kNN neighbors** → they are contradictory supervision on the boundary. Removing them un-warps the boundary → the many test points that also live near the boundary stop being random.
- **Measures:** margin & prob distribution of ambiguous-benign vs clean-benign vs malicious; kNN label-agreement per group; before/after cleaning, boundary stability + test recall near the boundary.

### Probe 3 — Loss interventions (asymmetric / focal / a new term)
- **Question:** what does asymmetric/focal loss actually *do* to the boundary?
- **Hypothesis H3:** it re-weights the majority so the boundary is pushed back toward benign; malicious margins increase. **Exploratory extension:** if the probe reveals a *specific* failure (e.g., a malicious sub-cluster consistently on the wrong side, or ambiguous-region collapse), design a **targeted loss term** for it and test empirically. High-risk / high-hanging fruit — attempt only if the data shows a clear, nameable pattern.
- **Measures:** margin distribution and recall@FPR under {no reweight, asymmetric, focal, new}; does the boundary move where the probe predicted?

### Probe 4 — OOD (subsection, supporting axis)
- **Question:** on a **new public malicious dataset**, does performance hold, and where do those prompts land in representation space?
- **Hypothesis H4:** a chunk lands in low-density / off-manifold regions (near-OOD) and gets small margins → the model extrapolates. Cross-dataset (LODO-style) drop quantifies it.
- **Measures:** perf drop cross-dataset (recall@FPR, PR-AUC); Mahalanobis / kNN distance of OOD-malicious to training distribution vs in-dist malicious.
- **Positioning:** cite Fomin (LODO) and Iyer 2026 (leave-one-family-out); we *adopt* their protocol as rigor, we don't claim OOD novelty.

## 3. First principles to own (also = research-engineer interview prep)

1. **The head is just a cut.** For a linear classification head, the decision boundary is a **hyperplane** in embedding space; the model's power lives in the **representation**. "Position in representation space" is literal. `margin(x) = signed distance to boundary ∝ logit`; `prob ≈ 0.5 ⇔ on the boundary`.
2. **UMAP / t-SNE preserve *local* structure, DISTORT *global* geometry.** Cluster sizes and inter-cluster distances are **not** trustworthy (t-SNE especially; UMAP a bit better but still). **Never quote a distance/gap from a projection as evidence.** Use the plot to *see / hypothesize*, then **confirm quantitatively in the original space.** ← the #1 interview gotcha.
3. **Quantitative probes are the real evidence:**
   - **Linear-probe accuracy** (logistic reg on frozen embeddings) → how linearly separable are the classes.
   - **Mahalanobis distance** to a class (accounts for covariance; assumes ~Gaussian class) → OOD / far-ness.
   - **kNN label agreement** → boundary/ambiguity (mixed neighbors = ambiguous).
   - **Silhouette** → cluster tightness/separation in the ORIGINAL space.
   - **Margin (logit) distribution** → where points sit relative to the boundary.
4. **OOD-ness ≠ softmax confidence.** Softmax is miscalibrated (over-confident OOD). Use distance/energy: **Mahalanobis, kNN-OOD, energy score, MSP baseline.**
5. **Imbalance changes the boundary, not the metric only.** At 1000:1 the loss is dominated by benign → boundary shifts into malicious → recall dies. PR-AUC / recall@fixed-FPR are the honest metrics; ROC-AUC hides it.

## 4. Likely interview questions (research engineer)
- t-SNE vs UMAP: what does each preserve/distort? Can you trust cluster distances? (→ no)
- How do you *verify* what you see in a projection isn't a hyperparameter artifact?
- How do you detect OOD without labels, and why not softmax confidence?
- Mahalanobis vs kNN-OOD — what assumption does Mahalanobis make? (Gaussian per class)
- Why does class imbalance hurt, and what do asymmetric/focal loss do to the boundary?
- PR-AUC vs ROC-AUC under imbalance; recall @ fixed low FPR.
- What is a linear probe and what does it (not) tell you?

## 5. Execution order (today → tomorrow)
- **Today (no retraining):** run `extract_embeddings.py` on the *existing trained model* over groups {clean-benign, malicious, ambiguous-benign, OOD-malicious}; then `probe.py` → quantitative table + UMAP/t-SNE + margin plots. This alone tests H2 and H4 and produces slide figures.
- **Tomorrow (needs training):** re-train at a couple of ratios and losses → repeat probe → tests H1, H3. Contrastive/new-loss only if the probe shows a clear, nameable pattern worth targeting.
