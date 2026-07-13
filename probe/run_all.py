#!/usr/bin/env python3
"""
run_all.py  —  batch-run every probe the stored embeddings support, in ONE launch.

Covers the experiments in final-presentation/experiments.md (Exp-1 metrics,
Exp-2 OOD both directions, Exp-2b ambiguous-benign, Exp-3 geometry + C<A,
Exp-4 calibration, coverage sub-probe). Missing inputs are SKIPPED with a clear
message, never crash the batch.

Input = Arrow shards. Each CONFIG entry points at a directory of `*.arrow` shards
(or a single .arrow file) holding the embeddings + inference results, one row per
prompt, with these columns (rename in COLS if yours differ):
    embedding   list<float> length D=384   pooled mmBERT-small masked-mean
    label       int                        0 = benign, 1 = malicious
    prob        float                       sigmoid(z/T)  (for AUPRC / recall@FPR)
    logit       float                       raw logit z   (for margins / calibration)

Fill CONFIG with your shard dirs. Key naming:
    {encoder-model}__{dataset}_{split}      e.g.  p5m__p5_test  =  PoC5-model encoding PoC5-test
The first load of each key prints its column names + row count for verification.
Run:  python run_all.py
"""
import os, sys, glob, warnings
from datetime import date
import numpy as np
import pyarrow as pa
import pyarrow.ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG — fill in the paths you saved. Leave a value as None if you don't have it.
# ----------------------------------------------------------------------------
CONFIG = {
    # each value = a directory of *.arrow shards (or a single .arrow file)
    # in-distribution (each split under its OWN model)
    "p5m__p5_train": "shards/p5model_p5train",
    "p5m__p5_test":  "shards/p5model_p5test",
    "p6m__p6_train": "shards/p6model_p6train",
    "p6m__p6_test":  "shards/p6model_p6test",
    # cross encodings (from the 2x2 inference)
    "p5m__p6_test":  "shards/p5model_p6test",    # old model -> new data  (cell B)
    "p6m__p5_test":  "shards/p6model_p5test",    # new model -> old data  (cell C)
    # calibration sets (logit + label is enough) -> Exp-4
    "p5m__p5_cal":   "shards/p5model_p5cal",
    "p6m__p6_cal":   "shards/p6model_p6cal",
    # optional: coverage sub-probe needs PoC6-train under the PoC5-model (extra pass)
    "p5m__p6_train": None,
}
# Arrow column names — rename to match your schema
COLS = {"emb": "embedding", "label": "label", "prob": "prob", "logit": "logit"}
# deployed calibration (for Exp-4 reproduction check)
DEPLOYED_T = {"p5m__p5_cal": 0.736, "p6m__p6_cal": 1.2811}
OUTDIR = os.path.join("results", date.today().isoformat())   # results/YYYY-MM-DD/
K_NN   = 10          # neighbours for kNN label-agreement / distance
TARGET_RECALL = 0.90 # PM operating point: precision @ recall 90%
TARGET_FPR = 0.01    # (kept for reference; base-rate-invariant alternative)

# ----------------------------------------------------------------------------
# loading + metric helpers
# ----------------------------------------------------------------------------
def _read_arrow(path):
    """Read a directory of *.arrow shards (or one .arrow file) into a single table."""
    files = sorted(glob.glob(os.path.join(path, "*.arrow"))) if os.path.isdir(path) else [path]
    tabs = []
    for f in files:
        try:                                              # HF-datasets = IPC stream format
            with pa.memory_map(f, "r") as s:
                tabs.append(pa.ipc.open_stream(s).read_all())
        except Exception:                                 # fall back to IPC file format
            with pa.memory_map(f, "r") as s:
                tabs.append(pa.ipc.open_file(s).read_all())
    return pa.concat_tables(tabs) if tabs else None

_seen = set()
def load(key):
    """Return dict(X,y,prob,logit) from Arrow shards, or None if absent."""
    path = CONFIG.get(key)
    if not path or not os.path.exists(path):
        return None
    tbl = _read_arrow(path)
    if tbl is None:
        return None
    if key not in _seen:                                  # first load: show schema for verification
        print(f"  [load {key}] columns={tbl.column_names}  rows={tbl.num_rows}")
        _seen.add(key)
    cols, out = tbl.column_names, {}
    if COLS["emb"] in cols:
        out["X"] = np.asarray(tbl.column(COLS["emb"]).to_pylist(), dtype=np.float32)
    for name, k in [("label", "y"), ("prob", "prob"), ("logit", "logit")]:
        if COLS[name] in cols:
            out[k] = np.asarray(tbl.column(COLS[name]).to_pylist())
    return out

def recall_at_fpr(y, score, target_fpr=TARGET_FPR):
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0

def precision_at_recall(y, score, target_recall=TARGET_RECALL):
    """PM operating point: best precision achievable while recall >= target_recall."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, score)
    ok = rec >= target_recall
    return float(prec[ok].max()) if ok.any() else 0.0

def linear_probe_acc(X, y):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return float(cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean())

def knn_label_agreement(X, y, k=K_NN):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)   # +1 to drop self
    _, idx = nn.kneighbors(X)
    neigh = y[idx[:, 1:]]
    return float((neigh == y[:, None]).mean())

def knn_agreement_perpoint(X, y, k=K_NN):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return (y[idx[:, 1:]] == y[:, None]).mean(axis=1)   # per-point fraction of same-label neighbours

def fit_class_gaussians(X, y, ridge=1e-3):
    stats = {}
    for c in np.unique(y):
        Xc = X[y == c]
        mu = Xc.mean(0)
        inv = np.linalg.pinv(np.cov(Xc.T) + ridge * np.eye(X.shape[1]))
        stats[c] = (mu, inv)
    return stats

def maha_min(X, stats):
    """closest-class Mahalanobis (Lee et al. 2018); higher = more OOD."""
    dists = []
    for mu, inv in stats.values():
        d = X - mu
        dists.append(np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d)))
    return np.min(np.stack(dists, 1), axis=1)

def knn_dist(Xq, Xref, k=5):
    nn = NearestNeighbors(n_neighbors=k).fit(Xref)
    d, _ = nn.kneighbors(Xq)
    return d.mean(1)

def bce_nll(logit, y, T):
    p = np.clip(1.0 / (1.0 + np.exp(-logit / T)), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def fit_temperature(logit, y):
    """T that minimises BCE-NLL (calibration). Grid + local refine, no scipy."""
    Ts = np.logspace(-1, 1, 400)          # 0.1 .. 10
    return float(Ts[int(np.argmin([bce_nll(logit, y, T) for T in Ts]))])

def ece(prob, y, n_bins=15):
    """Expected calibration error of the malicious-prob against the actual malicious rate."""
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(y)
    for i in range(n_bins):
        m = (prob >= bins[i]) & (prob < bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / N) * abs(prob[m].mean() - y[m].mean())
    return float(e)

def hdr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Experiment 1 — 2x2 metrics (AUPRC + recall@1%FPR)
# ----------------------------------------------------------------------------
hdr("Experiment 1 — 2x2 cross-evaluation (AUPRC + precision@recall%.0f%%)" % (TARGET_RECALL * 100))
cells = {"A": "p5m__p5_test", "B": "p5m__p6_test", "C": "p6m__p5_test", "D": "p6m__p6_test"}
print(f"{'cell':<6}{'set':<16}{'AUPRC':>10}{'prec@rec90':>13}")
for name, key in cells.items():
    d = load(key)
    if d is None:
        print(f"{name:<6}{key:<16}{'SKIP (no file)':>23}"); continue
    score = d["prob"] if "prob" in d else d["logit"]
    ap = average_precision_score(d["y"], score)
    pr = precision_at_recall(d["y"], score)
    print(f"{name:<6}{key:<16}{ap:>10.4f}{pr:>13.4f}")

# ----------------------------------------------------------------------------
# Experiment 3 — geometry per available set (linear probe / silhouette / kNN agree)
# ----------------------------------------------------------------------------
hdr("Experiment 3 — representation geometry per set")
print(f"{'set':<18}{'lin-probe':>11}{'silhouette':>12}{'kNN-agree':>11}{'N':>9}")
for key in CONFIG:
    d = load(key)
    if d is None:
        continue
    X, y = d["X"], d["y"]
    if len(np.unique(y)) < 2:
        continue
    try:
        lp = linear_probe_acc(X, y)
        sil = silhouette_score(X, y) if len(y) <= 20000 else float("nan")  # silhouette is O(N^2)
        agr = knn_label_agreement(X, y) if len(y) <= 50000 else float("nan")
        print(f"{key:<18}{lp:>11.4f}{sil:>12.4f}{agr:>11.4f}{len(y):>9}")
    except Exception as e:
        print(f"{key:<18}  error: {e}")

# ----------------------------------------------------------------------------
# Experiment 2 — OOD in representation space (forward + reverse)
# ----------------------------------------------------------------------------
def ood_direction(name, ref_key, indist_key, ood_key):
    ref, ind, ood = load(ref_key), load(indist_key), load(ood_key)
    if ref is None or ind is None or ood is None:
        print(f"[{name}] SKIP — need {ref_key}, {indist_key}, {ood_key}")
        return
    stats = fit_class_gaussians(ref["X"], ref["y"])
    md_in,  md_ood  = maha_min(ind["X"], stats), maha_min(ood["X"], stats)
    kd_in,  kd_ood  = knn_dist(ind["X"], ref["X"]), knn_dist(ood["X"], ref["X"])
    y_true  = np.r_[np.zeros(len(md_in)), np.ones(len(md_ood))]
    auroc_md = roc_auc_score(y_true, np.r_[md_in, md_ood])
    auroc_kd = roc_auc_score(y_true, np.r_[kd_in, kd_ood])
    print(f"[{name}]  AUROC(maha)={auroc_md:.3f}  AUROC(kNN)={auroc_kd:.3f}")
    print(f"         median maha  in={np.median(md_in):.2f}  ood={np.median(md_ood):.2f}")
    print(f"         median kNN   in={np.median(kd_in):.2f}  ood={np.median(kd_ood):.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, (t, si, so) in zip(ax, [("Mahalanobis", md_in, md_ood), ("kNN distance", kd_in, kd_ood)]):
        a.hist(si, bins=40, alpha=0.6, density=True, label="in-dist")
        a.hist(so, bins=40, alpha=0.6, density=True, label="OOD")
        a.axvline(np.quantile(si, 0.95), color="k", ls="--", lw=1, label="in-dist 95th pct")
        a.set_title(f"{name}: {t}"); a.legend()
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/ood_{name}.png", dpi=200)
    print(f"         wrote {OUTDIR}/ood_{name}.png")

hdr("Experiment 2 — OOD forward (is PoC6-test off the PoC5-train manifold?)")
ood_direction("forward", "p5m__p5_train", "p5m__p5_test", "p5m__p6_test")

hdr("Experiment 2 — OOD reverse (is PoC5-test off the PoC6-train manifold? explains C<A)")
ood_direction("reverse", "p6m__p6_train", "p6m__p6_test", "p6m__p5_test")

# ----------------------------------------------------------------------------
# Experiment 3b — C<A mechanism: PoC5-test separability under BOTH models
# ----------------------------------------------------------------------------
hdr("Experiment 3b — C<A mechanism: PoC5-test separability under both models")
for tag, key in [("PoC5-test @ PoC5-model", "p5m__p5_test"),
                 ("PoC5-test @ PoC6-model", "p6m__p5_test")]:
    d = load(key)
    if d is None:
        print(f"  {tag:<26} SKIP (no file)"); continue
    X, y = d["X"], d["y"]
    lp = linear_probe_acc(X, y)
    sil = silhouette_score(X, y) if len(y) <= 20000 else float("nan")
    print(f"  {tag:<26} lin-probe={lp:.4f}  silhouette={sil:.4f}")
print("  Hypothesis: PoC5-test is LESS separable under PoC6-model -> capacity spread -> explains C<A")

# ----------------------------------------------------------------------------
# Experiment 2b — ambiguous-benign proxy (H2 without set-ops)
#   Flag benign whose malicious-prob sits near the NATURAL boundary (|p-0.5| small),
#   and/or whose neighbours are label-mixed (low kNN agreement). Saves their indices.
# ----------------------------------------------------------------------------
hdr("Experiment 2b — ambiguous-benign proxy (H2, no set-ops)")
for key in ["p5m__p5_test", "p6m__p6_test"]:
    d = load(key)
    if d is None or "prob" not in d:
        print(f"  {key}: SKIP (need prob)"); continue
    ben = d["y"] == 0
    if ben.sum() == 0:
        print(f"  {key}: no benign"); continue
    prob_b = d["prob"][ben]
    near = np.abs(prob_b - 0.5) < 0.1                       # near natural boundary = intrinsic ambiguity
    ambiguous = near.copy()
    line = f"  {key}: benign={ben.sum()}  near-boundary|p-0.5|<0.1={near.sum()} ({100*near.mean():.2f}%)"
    if ben.sum() <= 50000:                                  # kNN feasible only on smaller sets
        agree = knn_agreement_perpoint(d["X"], d["y"])[ben]
        mixed = agree < 0.6
        ambiguous = near | mixed
        line += f"  mixed-kNN<0.6={mixed.sum()}  either={ambiguous.sum()}"
    else:
        line += "  (kNN skipped: benign set too large; prob-only)"
    print(line)
    np.save(f"{OUTDIR}/ambiguous_benign_idx__{key}.npy", np.where(ben)[0][ambiguous])
    plt.figure(figsize=(7, 4))
    plt.hist(prob_b, bins=60, density=True)
    plt.axvspan(0.4, 0.6, color="orange", alpha=0.3, label="ambiguous band")
    plt.xlabel("benign malicious-prob"); plt.title(f"{key}: benign prob (near 0.5 = ambiguous)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/ambiguous_benign__{key}.png", dpi=200)
    print(f"         wrote idx + {OUTDIR}/ambiguous_benign__{key}.png")

# ----------------------------------------------------------------------------
# Coverage sub-probe (optional) — which PoC6-train points are OOD vs PoC5-train
# ----------------------------------------------------------------------------
hdr("Coverage sub-probe — PoC6-train distance to PoC5-train manifold (curation lead)")
ref, p6tr = load("p5m__p5_train"), load("p5m__p6_train")
if ref is None or p6tr is None:
    print("  SKIP — needs p5m__p5_train AND p5m__p6_train (PoC6-train encoded by PoC5-model, extra pass)")
else:
    stats = fit_class_gaussians(ref["X"], ref["y"])
    md = maha_min(p6tr["X"], stats)
    thr = np.quantile(maha_min(ref["X"], stats), 0.95)     # in-dist 95th pct threshold
    frac_ood = float((md > thr).mean())
    print(f"  {frac_ood*100:.1f}% of PoC6-train is beyond PoC5-train's 95th-pct Mahalanobis")
    print(f"  -> that far tail is the 'new coverage' PoC6 added (candidate curation set)")
    plt.figure(figsize=(7, 4))
    plt.hist(maha_min(ref["X"], stats), bins=40, alpha=0.6, density=True, label="PoC5-train (ref)")
    plt.hist(md, bins=40, alpha=0.6, density=True, label="PoC6-train")
    plt.axvline(thr, color="k", ls="--", lw=1, label="ref 95th pct")
    plt.xlabel("closest-class Mahalanobis to PoC5-train"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/coverage_poc6train_vs_poc5train.png", dpi=200)
    print(f"  wrote {OUTDIR}/coverage_poc6train_vs_poc5train.png")

# ----------------------------------------------------------------------------
# Experiment 4 — Calibration: reproduce T/tau + does calibration transfer to OOD?
# ----------------------------------------------------------------------------
hdr("Experiment 4 — Calibration (reproduce T + does it transfer to OOD?)")
# (a) reproduce the deployed T on each cal set
for key, kT in DEPLOYED_T.items():
    d = load(key)
    if d is None:
        print(f"  reproduce T: SKIP {key}"); continue
    Tf = fit_temperature(d["logit"], d["y"])
    tau = 1.0 / (1.0 + np.exp(-np.quantile(d["logit"][d["y"] == 0], 0.99) / Tf))  # ~1% FPR cut on cal
    flag = "OK" if abs(Tf - kT) < 0.15 else "CHECK"
    print(f"  {key}: fitted T={Tf:.3f} (deployed {kT})  {flag} | tau@~1%FPR(cal)={tau:.3f}")

# (b) calibration transfer: in-dist (cal) T vs OOD-test T, same model
#     OOD set for PoC5-model = PoC6-test (cell B);  for PoC6-model = PoC5-test (cell C)
for tag, cal_key, ood_key in [("PoC5-model", "p5m__p5_cal", "p5m__p6_test"),
                              ("PoC6-model", "p6m__p6_cal", "p6m__p5_test")]:
    cal, ood = load(cal_key), load(ood_key)
    if cal is None or ood is None:
        print(f"  transfer {tag}: SKIP (need {cal_key} + {ood_key})"); continue
    T_in, T_ood = fit_temperature(cal["logit"], cal["y"]), fit_temperature(ood["logit"], ood["y"])
    p_dep = 1.0 / (1.0 + np.exp(-ood["logit"] / T_in))
    print(f"  {tag}: T_in(cal)={T_in:.3f}  T_ood={T_ood:.3f}  "
          f"NLL_ood@T_in={bce_nll(ood['logit'], ood['y'], T_in):.3f}  "
          f"NLL_ood@T_ood={bce_nll(ood['logit'], ood['y'], T_ood):.3f}  "
          f"ECE_ood@T_in={ece(p_dep, ood['y']):.3f}")
print("  Hypothesis: T_ood drifts from T_in => calibration does NOT transfer across the shift (OOD mis-calibration)")

hdr("DONE — fill the numbers above into experiments.md (Exp-1/2/3/4 tables)")
