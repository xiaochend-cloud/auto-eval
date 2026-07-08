"""
probe.py  —  open the black box.  Load embeddings.npz, run QUANTITATIVE probes FIRST,
then visualize.  Discipline: the plot generates hypotheses; the numbers are the evidence.

python probe.py     ->  prints a metrics table, writes  probe_umap.png / probe_margins.png
"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

d = np.load("embeddings.npz", allow_pickle=True)
X, y, group, prob, logit = d["X"], d["y"], d["group"], d["prob"], d["logit"]
groups = list(dict.fromkeys(group.tolist()))
print(f"N={len(y)}  D={X.shape[1]}  groups={groups}\n")

# ---------- 1. Linear-probe separability (how good is the representation?) ----------
# High CV acc => classes are linearly separable in embedding space (head can work).
in_dist = np.isin(group, ["clean_benign", "malicious"])
acc = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                      X[in_dist], y[in_dist], cv=5, scoring="balanced_accuracy").mean()
print(f"[linear-probe] balanced acc (benign vs malicious): {acc:.3f}")
print(f"[silhouette]  benign vs malicious (orig space):    {silhouette_score(X[in_dist], y[in_dist]):.3f}\n")

# ---------- 2. kNN label agreement: ambiguity / boundary test (Hypothesis H2) ----------
# For each point, do its k nearest neighbors share its label? Mixed neighbors = ambiguous.
k = 10
nn = NearestNeighbors(n_neighbors=k+1).fit(X)
_, idx = nn.kneighbors(X)
neigh_lab = y[idx[:, 1:]]                       # drop self
agree = (neigh_lab == y[:, None]).mean(axis=1)  # fraction of same-label neighbors
print("[kNN label agreement]  (lower = more boundary/ambiguous)")
for g in groups:
    m = group == g
    print(f"   {g:18s}  agree={agree[m].mean():.3f}   |margin|median={np.median(np.abs(logit[m])):.2f}   prob_mean={prob[m].mean():.3f}")

# ---------- 3. Mahalanobis distance to training MALICIOUS region (OOD test, H4) ----------
mal = (group == "malicious")
mu  = X[mal].mean(0)
cov = np.cov(X[mal].T) + 1e-3*np.eye(X.shape[1])
inv = np.linalg.pinv(cov)
def maha(A):
    diff = A - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
print("\n[Mahalanobis to in-dist malicious]  (higher = more OOD / off-manifold)")
for g in groups:
    m = group == g
    print(f"   {g:18s}  maha_median={np.median(maha(X[m])):.2f}")

# ---------- 4. Margin distribution per group (boundary picture, H1/H2/H3) ----------
plt.figure(figsize=(7,4))
for g in groups:
    m = group == g
    plt.hist(logit[m], bins=40, alpha=0.5, density=True, label=g)
plt.axvline(0, color="k", lw=1, ls="--")           # the decision boundary
plt.xlabel("signed margin (logit; >0 => predicted malicious)"); plt.ylabel("density")
plt.title("Where each group sits relative to the decision boundary"); plt.legend()
plt.tight_layout(); plt.savefig("probe_margins.png", dpi=200)
print("\nwrote probe_margins.png")

# ---------- 5. UMAP — SEE structure only. DO NOT quote distances from this plot. ----------
try:
    import umap
    emb2 = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(13,5))
    for g in groups:
        m = group == g
        ax[0].scatter(emb2[m,0], emb2[m,1], s=6, alpha=0.5, label=g)
    ax[0].legend(); ax[0].set_title("UMAP by group  (topology, NOT distances)")
    sc = ax[1].scatter(emb2[:,0], emb2[:,1], c=prob, s=6, cmap="coolwarm")
    plt.colorbar(sc, ax=ax[1]); ax[1].set_title("UMAP colored by malicious prob")
    plt.tight_layout(); plt.savefig("probe_umap.png", dpi=200)
    print("wrote probe_umap.png   (reminder: distances in UMAP are NOT trustworthy)")
except ImportError:
    print("umap-learn not installed; skip viz (pip install umap-learn)")
