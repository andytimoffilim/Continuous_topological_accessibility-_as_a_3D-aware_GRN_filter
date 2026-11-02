# C:\Users\user\PycharmProjects\TAD\step2_operator_audit.py
# Operator audit for RFA: P correctness, A symmetry, gamma->1 limit,
# variance contraction, PC1 residualization ratio.
# Writes: E:\HIC\results\operator_audit\operator_audit.csv
#         E:\HIC\results\operator_audit\operator_audit.txt
# Does NOT recompute P. Does NOT silently "fix" mismatches.

import os
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import spsolve

# ---------------- paths ----------------
HIC_CHR1_10KB = r"E:\HIC\results\GM12878_chr1_10kb.npy"
P_CHR1        = r"E:\HIC\results\pilot_multitf\P_matrix.npz"
A_CHR1        = r"E:\HIC\results\pilot_multitf\A_matrix.npz"
PC1_CHR1      = r"E:\HIC\results\pilot_multitf\pc1.npy"

P_CHR17       = r"E:\HIC\results\eqtl_multi_chr\P_chr17.npz"
A_CHR17       = r"E:\HIC\results\eqtl_multi_chr\A_chr17.npz"
PC1_CHR17     = r"E:\HIC\results\eqtl_multi_chr\pc1_chr17.npy"

P_CHR19       = r"E:\HIC\results\eqtl_multi_chr\P_chr19.npz"
A_CHR19       = r"E:\HIC\results\eqtl_multi_chr\A_chr19.npz"
PC1_CHR19     = r"E:\HIC\results\eqtl_multi_chr\pc1_chr19.npy"

TFBS_BED      = r"E:\genomes\tfbs_GM12878.bed"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"

OUT_DIR       = r"E:\HIC\results\operator_audit"
OUT_CSV       = os.path.join(OUT_DIR, "operator_audit.csv")
OUT_TXT       = os.path.join(OUT_DIR, "operator_audit.txt")
os.makedirs(OUT_DIR, exist_ok=True)

TOPK          = 300
GAMMA         = 0.85
GAMMA_GRID    = [0.5, 0.7, 0.85, 0.95, 0.99, 0.999]
TOL_ROWSUM    = 1e-8
TOL_SYM       = 1e-12

# ---------------- helpers ----------------
def load_P(path):
    M = load_npz(path)
    return csr_matrix(M)

def load_A(path):
    M = load_npz(path)
    return csr_matrix(M)

def rebuild_A_raw(hic_npy_path, topk=300):
    """Rebuild raw A from .npy Hi-C: diag=0, top-k per row, max(A,A^T)."""
    C = np.load(hic_npy_path)
    assert C.ndim == 2 and C.shape[0] == C.shape[1]
    n = C.shape[0]
    A = np.zeros_like(C, dtype=np.float64)
    for i in range(n):
        row = C[i].astype(np.float64).copy()
        row[i] = 0.0
        if row.sum() == 0:
            continue
        # top-k by value
        if topk < n - 1:
            idx = np.argpartition(row, -topk)[-topk:]
        else:
            idx = np.arange(n)
        A[i, idx] = row[idx]
    A = np.maximum(A, A.T)
    return A

def check_P(P, name, tol=1e-8):
    P = csr_matrix(P)
    n = P.shape[0]
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    max_dev = float(np.max(np.abs(rowsum - 1.0)))
    n_bad = int(np.sum(np.abs(rowsum - 1.0) > tol))
    Pmin = float(P.data.min()) if P.nnz > 0 else 0.0
    nnz = int(P.nnz)
    deg = rowsum
    return dict(
        name=name, n=n, nnz=nnz, P_min=Pmin,
        rowsum_max_dev=max_dev, n_rows_bad=n_bad,
        rowsum_ok=(max_dev <= tol), P_nonneg=(Pmin >= 0.0),
        deg_min=float(deg.min()), deg_med=float(np.median(deg)), deg_max=float(deg.max()),
    )

def check_A_sym(A, name, tol=1e-12):
    A = csr_matrix(A)
    diff = A - A.T
    max_asym = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
    return dict(name=name, n=A.shape[0], nnz=int(A.nnz),
                max_asym=max_asym, symmetric=(max_asym <= tol))

def gamma_limit_check(P, v0, gamma_grid, name):
    """
    For gamma -> 1: (1-gamma)(I - gamma P)^{-1} v0
    Correct limit: (pi . v0 / pi . 1) * 1, where pi is stationary dist of P.
    We check convergence to this limit and NOT to pi itself.
    """
    n = P.shape[0]
    # stationary pi: left eigenvector for eigenvalue 1 of P
    # pi solves pi P = pi, sum pi = 1. Equivalent: P^T pi = pi.
    # Use power iteration on P^T (row-stochastic => pi = deg / sum(deg) for undirected).
    # For a general row-stochastic P from symmetric A, pi_i = d_i / sum(d).
    d = np.asarray(P.sum(axis=1)).ravel()  # rowsum == 1 here; use degrees from A instead
    # We need degrees of A, not rowsum of P. Recover: d_i = 1 / P_ii? No.
    # Use: pi is stationary iff pi P = pi. Compute by power iteration on P^T.
    pi = np.ones(n) / n
    for _ in range(1000):
        pi_new = P.T @ pi
        pi_new = pi_new / pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < 1e-14:
            pi = pi_new
            break
        pi = pi_new
    pi_dot_v0 = float(pi @ v0)
    pi_dot_1  = float(pi @ np.ones(n))
    target_const = pi_dot_v0 / pi_dot_1  # scalar; limit vector = target_const * 1

    rows = []
    for g in gamma_grid:
        I = csr_matrix(np.eye(n))
        M = (I - g * P).tocsc()
        x = spsolve(M, v0)
        scaled = (1.0 - g) * x
        # compare to constant vector
        const_est = float(np.mean(scaled))
        dev_from_const = float(np.max(np.abs(scaled - const_est)))
        dev_from_target = float(np.max(np.abs(scaled - target_const)))
        dev_from_pi = float(np.max(np.abs(scaled - pi)))
        rows.append(dict(
            name=name, gamma=g,
            mean_scaled=const_est,
            dev_from_const=dev_from_const,
            dev_from_target=dev_from_target,
            dev_from_pi=dev_from_pi,
            target_const=target_const,
        ))
    return rows, target_const, pi

def variance_contraction(P, v0, gamma_grid, name):
    """Empirical Var[x_gamma] vs Var[v0] on bin level."""
    n = P.shape[0]
    var_v0 = float(np.var(v0))
    rows = []
    for g in gamma_grid:
        I = csr_matrix(np.eye(n))
        M = (I - g * P).tocsc()
        x = spsolve(M, v0)
        rows.append(dict(
            name=name, gamma=g,
            var_v0=var_v0, var_x=float(np.var(x)),
            ratio=float(np.var(x) / var_v0) if var_v0 > 0 else np.nan,
            contracted=(np.var(x) < var_v0),
        ))
    return rows

def pc1_resid_ratio(P, pc1, v0, name):
    """
    Definition: rfa_pc1_resid_ratio = Var(residual of RFA on PC1) / Var(RFA),
    computed at gene level (bins that contain a promoter).
    We use ALL bins as a proxy if gene annotation is unavailable; but we have
    GENE_STRUCT, so use real promoters.
    """
    import pandas as pd
    gs = pd.read_csv(GENE_STRUCT)
    gs = gs[gs["element_type"] == "promoter"].copy()
    gs["chrom"] = gs["chrom"].astype(str).str.replace("chr", "", regex=False)
    gs = gs[gs["chrom"] == "1"]
    gs = gs.drop_duplicates(subset=["gene_id"], keep="first")
    # bin = floor(start / 10000)
    bins = (gs["start"].astype(int) // 10000).values
    bins = bins[(bins >= 0) & (bins < P.shape[0])]
    bins = np.unique(bins)
    if len(bins) < 50:
        return dict(name=name, n_genes=len(bins), ratio=np.nan, note="too few genes")

    I = csr_matrix(np.eye(P.shape[0]))
    M = (I - GAMMA * P).tocsc()
    x = spsolve(M, v0)  # full RFA profile
    x_g = x[bins]
    pc1_g = pc1[bins]
    # OLS: x_g ~ pc1_g
    X = np.column_stack([np.ones_like(pc1_g), pc1_g])
    beta, *_ = np.linalg.lstsq(X, x_g, rcond=None)
    resid = x_g - X @ beta
    ratio = float(np.var(resid) / np.var(x_g)) if np.var(x_g) > 0 else np.nan
    return dict(name=name, n_genes=len(bins), ratio=ratio, note="ok")

# ---------------- run ----------------
report_lines = []
def log(s):
    print(s)
    report_lines.append(s)

log("=" * 80)
log("OPERATOR AUDIT — RFA")
log("=" * 80)

# --- 1. Load cache ---
log("\n[1] Loading cached P / A / PC1 ...")
P1 = load_P(P_CHR1); A1 = load_A(A_CHR1); pc1_1 = np.load(PC1_CHR1)
P17 = load_P(P_CHR17); A17 = load_A(A_CHR17); pc1_17 = np.load(PC1_CHR17)
P19 = load_P(P_CHR19); A19 = load_A(A_CHR19); pc1_19 = np.load(PC1_CHR19)
C1 = np.load(HIC_CHR1_10KB)
log(f"  chr1 : P {P1.shape}, A {A1.shape}, pc1 {pc1_1.shape}, hic {C1.shape}")
log(f"  chr17: P {P17.shape}, A {A17.shape}, pc1 {pc1_17.shape}")
log(f"  chr19: P {P19.shape}, A {A19.shape}, pc1 {pc1_19.shape}")

# --- 2. P correctness ---
log("\n[2] P correctness (stochastic, nonneg, zero rows) ...")
p_rows = []
for P, nm in [(P1, "chr1"), (P17, "chr17"), (P19, "chr19")]:
    r = check_P(P, nm)
    p_rows.append(r)
    log(f"  {nm}: n={r['n']}, nnz={r['nnz']}, P_min={r['P_min']:.3e}, "
        f"rowsum_max_dev={r['rowsum_max_dev']:.3e}, n_bad={r['n_rows_bad']}, "
        f"stochastic={r['rowsum_ok']}, nonneg={r['P_nonneg']}")
    log(f"       deg: min={r['deg_min']:.3e} med={r['deg_med']:.3e} max={r['deg_max']:.3e}")

# --- 3. A symmetry: cached vs raw ---
log("\n[3] A symmetry: cached A_matrix.npz vs raw rebuilt from .npy ...")
a_rows = []
for A, nm in [(A1, "chr1_cached"), (A17, "chr17_cached"), (A19, "chr19_cached")]:
    r = check_A_sym(A, nm)
    a_rows.append(r)
    log(f"  {nm}: n={r['n']}, nnz={r['nnz']}, max|A-A^T|={r['max_asym']:.3e}, "
        f"symmetric={r['symmetric']}")

log("  rebuilding A_raw from GM12878_chr1_10kb.npy (top-k=300, sym, diag=0)...")
A1_raw = rebuild_A_raw(HIC_CHR1_10KB, TOPK)
r = check_A_sym(csr_matrix(A1_raw), "chr1_raw")
a_rows.append(r)
log(f"  chr1_raw: n={r['n']}, nnz={r['nnz']}, max|A-A^T|={r['max_asym']:.3e}, "
    f"symmetric={r['symmetric']}")

# compare cached A to raw A
diff = (A1 - csr_matrix(A1_raw))
max_diff = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
log(f"  max|A_cached - A_raw| = {max_diff:.3e}  "
    f"({'cached A is raw' if max_diff < 1e-12 else 'cached A is residualized/modified'})")

# --- 4. gamma -> 1 limit ---
log("\n[4] gamma -> 1 limit: (1-gamma)(I - gamma P)^{-1} v0 ...")
# v0 = TF peak indicator for chr1 (IKZF1 as example; full 20-TF would be slow)
import pandas as pd
tfbs = pd.read_csv(TFBS_BED, sep="\t", header=None,
                   names=["chrom","start","end","name","score","strand"])
tfbs["chrom"] = tfbs["chrom"].astype(str).str.replace("chr", "", regex=False)
ikzf1 = tfbs[(tfbs["chrom"] == "1") & (tfbs["name"].astype(str).str.contains("IKZF1", case=False, na=False))]
n_bins = P1.shape[0]
v0 = np.zeros(n_bins)
for _, row in ikzf1.iterrows():
    b = int(row["start"]) // 10000
    if 0 <= b < n_bins:
        v0[b] = 1.0
log(f"  v0 (IKZF1 chr1): n_peaks={int(v0.sum())}, n_bins={n_bins}")

lim_rows, target_const, pi = gamma_limit_check(P1, v0, GAMMA_GRID, "chr1_IKZF1")
log(f"  target const = (pi.v0)/(pi.1) = {target_const:.6e}")
for r in lim_rows:
    log(f"  gamma={r['gamma']:.3f}: mean_scaled={r['mean_scaled']:.6e}, "
        f"dev_from_const={r['dev_from_const']:.3e}, "
        f"dev_from_target={r['dev_from_target']:.3e}, "
        f"dev_from_pi={r['dev_from_pi']:.3e}")
log("  NOTE: limit is (pi.v0)/(pi.1) * 1, NOT pi. "
    "Check: dev_from_target should decrease as gamma->1.")

# --- 5. variance contraction ---
log("\n[5] Variance contraction: Var[x_gamma] vs Var[v0] ...")
var_rows = variance_contraction(P1, v0, GAMMA_GRID, "chr1_IKZF1")
for r in var_rows:
    log(f"  gamma={r['gamma']:.3f}: var_v0={r['var_v0']:.6e}, var_x={r['var_x']:.6e}, "
        f"ratio={r['ratio']:.4f}, contracted={r['contracted']}")
log("  NOTE: contraction is EMPIRICAL on this gamma grid, not a theorem. "
    "Text should say 'empirically contracts variance', not 'is a contraction'.")

# --- 6. PC1 residualization ratio ---
log("\n[6] rfa_pc1_resid_ratio (definition + match with 0.623) ...")
pc1r = pc1_resid_ratio(P1, pc1_1, v0, "chr1_IKZF1")
log(f"  definition: Var(residual of RFA on PC1)/Var(RFA), gene-level promoters chr1")
log(f"  n_genes={pc1r['n_genes']}, ratio={pc1r['ratio']:.4f}")
EXPECTED = 0.623
if pc1r['ratio'] is not np.nan and abs(pc1r['ratio'] - EXPECTED) < 0.05:
    log(f"  MATCH with expected {EXPECTED} (within 0.05)")
else:
    log(f"  MISMATCH: got {pc1r['ratio']:.4f}, expected {EXPECTED}. "
        f"DO NOT silently recompute — ask user.")

# --- write CSV ---
log("\n[7] Writing outputs ...")
import pandas as pd
pd.DataFrame(p_rows).to_csv(OUT_CSV.replace(".csv", "_P.csv"), index=False)
pd.DataFrame(a_rows).to_csv(OUT_CSV.replace(".csv", "_A.csv"), index=False)
pd.DataFrame(lim_rows).to_csv(OUT_CSV.replace(".csv", "_gamma_limit.csv"), index=False)
pd.DataFrame(var_rows).to_csv(OUT_CSV.replace(".csv", "_var.csv"), index=False)
pd.DataFrame([pc1r]).to_csv(OUT_CSV.replace(".csv", "_pc1.csv"), index=False)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"  wrote {OUT_CSV}* and {OUT_TXT}")

log("\n" + "=" * 80)
log("AUDIT DONE. Read operator_audit.txt and the 5 CSVs before step 3.")
log("=" * 80)