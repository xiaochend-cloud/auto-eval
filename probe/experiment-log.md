# repr-probing — working log

*Growing WIP log: data inventory, run status, partial results, decisions. The clean presentation-facing writeup lives in `presentation/final-presentation/experiments.md`. Batch runner: `probe/run_all.py`.*

---

## Data inventory (embeddings actually extracted)

Everything is computed offline from stored `{X = pooled embedding (mmBERT-small masked-mean, D=384), y = label, z = logit}` per (model × split).

**Have now — TEST** (all four 2×2 cells): `{X, y, prob, logit}` for PoC5/PoC6-model × PoC5/PoC6-test.
- Validated: class-Gaussian fit on ID malicious gives median MD ≈ √384 ≈ 19.6 (matches χ²₍₃₈₄₎) → saved embeddings + code are sound.
- Unlocks: Exp-1, Exp-3/3b, Exp-2b, and a first-pass Exp-2 (test-fit reference).

**To run next — TRAIN**: `p5m__p5_train`, `p6m__p6_train` → honest held-out ID reference for Exp-2 (fit μ,Σ on train, score test). `p5m__p6_train` (PoC6-train under PoC5-model) → coverage sub-probe.

**To run next — CAL**: `p5m__p5_cal`, `p6m__p6_cal` (logit + y suffice) → Exp-4 calibration.

**Out of the `{X,y,z}` closure (needs new training, not data-wrangling):** H1 (retrain at other ratios), H3 (retrain with other losses). H2 ground-truth set needs a version set-difference; Exp-2b approximates it now via near-boundary / mixed-neighbour benign.

---

## Run status

| block | needs | status |
|---|---|---|
| Exp-1 (2×2 metrics) | test prob/logit | ✅ AUPRC done; recall@1%FPR pending run_all |
| Exp-2 OOD (validation pass) | test-fit MD | ✅ ran on test malicious (below) |
| Exp-2 OOD (honest) | train embeddings | ⏳ need train |
| Exp-2b ambiguous-benign | test X,prob | ⏳ in run_all, not yet run |
| Exp-3 / 3b geometry | test X,y | ⏳ in run_all, not yet run |
| Exp-4 calibration | cal logit | ⏳ need cal |
| coverage | p5m__p6_train | ⏳ optional extra encoding |

---

## Results so far

### Exp-1 — 2×2 (AUPRC)
| | PoC5-test | PoC6-test |
|---|---|---|
| PoC5-model | A = 0.94 | B = 0.42 |
| PoC6-model | C = 0.77 | D = 0.985 |

Verdict: gain-from-coverage + mild specialization trade. B≪A and D≫B (PoC6 absorbed the region OOD for PoC5); C<A (small cost on PoC5's own region). Old = brittle specialist, new = robust generalist.

### Exp-2 validation — Mahalanobis, both directions (fit on **test** malicious; d=384 so ID ≈ √384 ≈ 19.6)

**Probe 1 — fit μ,Σ on PoC5-test malicious (PoC5-model embeddings):**
| Group | Median MD | Mean MD | Std |
|---|---|---|---|
| PoC5 test malicious (ID) | 19.36 | 19.38 | 2.86 |
| PoC6 test malicious (OOD) | 26.24 | 33.77 | 17 |
| PoC5 test benign | 50.91 | 53.13 | 14.01 |
| PoC6 test benign | 49.39 | 51.58 | 13.84 |

→ PoC6 malicious shifted out with a heavy tail (mean ≫ median) = **partially OOD** relative to the PoC5 malicious manifold.

**Probe 2 — fit μ,Σ on PoC6-test malicious (PoC6-model embeddings):**
| Group | Median MD | Mean MD | Std |
|---|---|---|---|
| PoC6 test malicious (ID) | 19.01 | 19.32 | 3.22 |
| PoC5 test malicious (OOD) | 19.60 | 19.94 | 3.43 |
| PoC6 test benign | 31.73 | 33.98 | 9.88 |
| PoC5 test benign | 31.71 | 34.18 | 10.58 |

→ PoC5 malicious is **not** OOD under PoC6 (same MD + std as ID, no tail).

**Finding (asymmetric containment):** the PoC6 malicious manifold **encompasses** PoC5 malicious; the reverse does not hold. Explains the 2×2 directionally.

**Note:** validation used a **test**-fit reference (fit + score on the same set for the ID row = slightly optimistic). Final honest run fits on **train** malicious, scores test.

---

## Open questions / next

- **C<A cause:** ruled out "malicious became OOD" (Probe 2 shows PoC5-mal contained). Leading hypothesis = tightness/boundary trade-off (broader malicious manifold → less tight around PoC5's region → smaller margin). **Confirm with Exp-3b** (PoC5-test separability under both models) + margin distribution.
- **Do next:** extract train + cal embeddings → run full `run_all.py` → backfill blanks in `experiments.md`.
- **Later (needs GPU/retrain):** H1 (ratios), H3 (losses).
