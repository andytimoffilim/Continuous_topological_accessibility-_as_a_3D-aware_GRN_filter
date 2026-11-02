# C:\Users\user\PycharmProjects\TAD\step3_5_topology_controls_chr17_19.py
# Step 3.5: chr17/19 replication of topology controls.
# Same pipeline as step3_topology_controls.py:
#   residualize on PC1 (same per-chrom pc1.npy), clip, self-loop on zero rows, P=D^-1 A.
# Real Hi-C vs 3 controls (degree-preserving rewiring, distance-preserving shuffle,
# flattened graph). Metrics: AUC(RFA), AUC(-RFA), rho(RFA, -log10 p_min),
# paired bootstrap CI on Delta vs real.
# Does NOT recompute P for chr1/17/19. Does NOT tune gamma.

import os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
CHROMS = ['chr17', 'chr19']

CHROM_CONFIG = {
    'chr17': dict(
        chrom_plain = '17',
        hic_npy     = r"E:\HIC\results\GM12878_chr17_10kb.npy",
        p_cache     = r"E:\HIC\results\eqtl_multi_chr\P_chr17.npz",
        pc1_cache   = r"E:\HIC\results\eqtl_multi_chr\pc1_chr17.npy",
        out_csv     = r"E:\HIC\results\topology_controls\topology_controls_chr17_5tf.csv",
        out_txt     = r"E:\HIC\results\topology_controls\topology_controls_chr17_5tf.txt",
    ),
    'chr19': dict(
        chrom_plain = '19',
        hic_npy     = r"E:\HIC\results\GM12878_chr19_10kb.npy",
        p_cache     = r"E:\HIC\results\eqtl_multi_chr\P_chr19.npz",
        pc1_cache   = r"E:\HIC\results\eqtl_multi_chr\pc1_chr19.npy",
        out_csv     = r"E:\HIC\results\topology_controls\topology_controls_chr19_5tf.csv",
        out_txt     = r"E:\HIC\results\topology_controls\topology_controls_chr19_5tf.txt",
    ),
}

BED_PATH      = r"E:\genomes\tfbs_GM12878.bed"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"
EQTL_TARGET   = r"E:\HIC\eqtl\eqtl_target.csv"
OUT_DIR       = r"E:\HIC\results\topology_controls"

RESOLUTION    = 10_000
CELL_LINE     = "GM12878"
GAMMA         = 0.85
TOPK          = 300
N_BOOT        = 1000
RNG_SEED      = 42
N_SWAP_FACTOR = 10   # <-- reduced from 10 (chr1 took 635s; 2x nnz ~ 1-2 min)

TFS = ['IKZF1', 'CTCF', 'EP300', 'POLR2A', 'YY1']

os.makedirs(OUT_DIR, exist_ok=True)


# ==================== I/O (per-chrom) ====================
def load_genes(chrom_plain):
    df = pd.read_csv(GENE_STRUCT, low_memory=False)
    df = df[(df['element_type'] == 'promoter') &
            (df['chrom'].astype(str) == chrom_plain)]
    df = df.sort_values(['gene_id', 'start']).drop_duplicates('gene_id', keep='first')
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    df['bin'] = (df['start'] // RESOLUTION).astype(int)
    return df[['gene_id', 'start', 'end', 'gene_name', 'bin']]


def load_eqtl():
    df = pd.read_csv(EQTL_TARGET, low_memory=False)
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    assert 'gene_id' in df.columns and 'min_p' in df.columns, df.columns.tolist()
    df['neg_log10_p'] = -np.log10(df['min_p'].clip(lower=1e-300))
    return df[['gene_id', 'min_p', 'neg_log10_p']]


def peaks_to_vector(bed_path, chrom, tf, n_bins):
    vec = np.zeros(n_bins, dtype=np.float64)
    if not os.path.exists(bed_path):
        return vec
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6: continue
            if p[0] != chrom or p[3] != tf: continue
            if CELL_LINE not in p[5]: continue
            b = int(p[1]) // RESOLUTION
            if 0 <= b < n_bins:
                vec[b] = 1.0
    return vec


# ==================== GRAPH BUILD (identical to step3) ====================
def rebuild_A_raw(hic_npy_path, topk=300):
    C = np.load(hic_npy_path)
    n = C.shape[0]
    A = np.zeros_like(C, dtype=np.float64)
    for i in range(n):
        row = C[i].astype(np.float64).copy()
        row[i] = 0.0
        if row.sum() == 0:
            continue
        if topk < n - 1:
            idx = np.argpartition(row, -topk)[-topk:]
        else:
            idx = np.arange(n)
        A[i, idx] = row[idx]
    A = np.maximum(A, A.T)
    return A


def residualize_on_pc1_sparse(A_sp, pc1):
    """Same logic as MULTI-TF PILOT.py: skip rows with <5 nnz, keep resid>0."""
    A_sp = A_sp.tocsr()
    n = A_sp.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        s, e = A_sp.indptr[i], A_sp.indptr[i+1]
        if s == e: continue
        ci, rv = A_sp.indices[s:e], A_sp.data[s:e]
        if len(ci) < 5:
            rows.extend([i]*len(ci)); cols.extend(ci.tolist()); vals.extend(rv.tolist())
            continue
        X = np.column_stack([np.ones(len(ci)), pc1[ci]])
        beta, *_ = np.linalg.lstsq(X, rv, rcond=None)
        resid = rv - X @ beta
        keep = resid > 0
        rows.extend([i]*keep.sum())
        cols.extend(ci[keep].tolist()); vals.extend(resid[keep].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def apply_self_loop(P):
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    zero = np.where(rowsum == 0)[0]
    if len(zero) > 0:
        P = P.tolil()
        for i in zero:
            P[i, i] = 1.0
        P = P.tocsr()
    return P, len(zero)


def build_P_from_A(A_sp, pc1):
    A_res = residualize_on_pc1_sparse(A_sp, pc1)
    row_sums = np.array(A_res.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = (D_inv @ A_res).tocsr()
    P, n_zero = apply_self_loop(P)
    return P, n_zero


# ==================== CONTROLS (identical to step3) ====================
def edges_from_A(A_dense):
    iu = np.triu_indices_from(A_dense, k=1)
    mask = A_dense[iu] > 0
    return np.column_stack([iu[0][mask], iu[1][mask]])


def double_edge_swap_numpy(A_dense, n_swap, seed=42):
    rng = np.random.default_rng(seed)
    n = A_dense.shape[0]
    edges = edges_from_A(A_dense).tolist()
    edge_set = set(map(tuple, edges))
    m = len(edges)
    swaps = 0
    attempts = 0
    max_attempts = n_swap * 100
    while swaps < n_swap and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, m, 2)
        if i == j: continue
        a, b = edges[i]
        c, d = edges[j]
        if a == c or a == d or b == c or b == d: continue
        if (a, d) in edge_set or (c, b) in edge_set: continue
        if a == d or c == b: continue
        edge_set.discard((a, b)); edge_set.discard((c, d))
        edge_set.add((a, d)); edge_set.add((c, b))
        edges[i] = (a, d); edges[j] = (c, b)
        swaps += 1
    A_new = np.zeros_like(A_dense)
    for (a, b) in edge_set:
        A_new[a, b] = A_new[b, a] = 1.0
    return A_new, swaps


def distance_preserving_shuffle(A_dense, seed=42):
    rng = np.random.default_rng(seed)
    n = A_dense.shape[0]
    A_new = np.zeros_like(A_dense)
    for d in range(1, n):
        diag = np.diagonal(A_dense, offset=d).copy()
        if diag.sum() == 0: continue
        rng.shuffle(diag)
        A_new[np.arange(n - d), np.arange(d, n)] = diag
    A_new = np.maximum(A_new, A_new.T)
    return A_new


def flattened_graph(A_dense, seed=42):
    rng = np.random.default_rng(seed)
    n = A_dense.shape[0]
    deg = (A_dense > 0).sum(axis=1).astype(int)
    stubs = []
    for i, d in enumerate(deg):
        stubs.extend([i] * int(d))
    stubs = np.array(stubs)
    rng.shuffle(stubs)
    A_new = np.zeros((n, n), dtype=np.float64)
    used = set()
    for k in range(0, len(stubs) - 1, 2):
        a, b = int(stubs[k]), int(stubs[k+1])
        if a == b: continue
        key = (min(a, b), max(a, b))
        if key in used: continue
        used.add(key)
        A_new[a, b] = A_new[b, a] = 1.0
    return A_new


# ==================== RFA (identical to step3) ====================
def rfa_profile(P, v0, gamma):
    n = P.shape[0]
    A_sys = (sp.eye(n, format='csr') - gamma * P).T.tocsc()
    v0 = np.asarray(v0, dtype=np.float64).ravel()
    try:
        ilu = spilu(A_sys, drop_tol=1e-5, fill_factor=20)
        M = LinearOperator(A_sys.shape, ilu.solve)
        x, info = gmres(A_sys, v0, M=M, rtol=1e-7, atol=0, maxiter=1000)
        if info != 0:
            x, info = gmres(A_sys, v0, M=M, rtol=1e-6, atol=0, maxiter=2000)
        return x
    except Exception:
        return spsolve(A_sys, v0)


# ==================== STATS (identical to step3) ====================
def paired_bootstrap_delta(y, score_real, score_ctrl, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d_obs = roc_auc_score(y, score_real) - roc_auc_score(y, score_ctrl)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            d = roc_auc_score(yt, score_real[idx]) - roc_auc_score(yt, score_ctrl[idx])
            deltas.append(d)
        except ValueError:
            continue
    if not deltas:
        return d_obs, np.nan, np.nan
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return d_obs, float(lo), float(hi)


def paired_bootstrap_delta_rho(x_real, x_ctrl, y, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d_obs = spearmanr(x_real, y).statistic - spearmanr(x_ctrl, y).statistic
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x_real[idx])) < 2 or len(np.unique(x_ctrl[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        d = spearmanr(x_real[idx], y[idx]).statistic - spearmanr(x_ctrl[idx], y[idx]).statistic
        if not np.isnan(d):
            deltas.append(d)
    if not deltas:
        return d_obs, np.nan, np.nan
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return d_obs, float(lo), float(hi)


def bootstrap_auc_ci(y, score, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if len(np.unique(yt)) < 2: continue
        aucs.append(roc_auc_score(yt, score[idx]))
    if not aucs: return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_rho_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2: continue
        r = spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(r): rhos.append(r)
    if not rhos: return np.nan, np.nan
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


# ==================== PER-CHROM PIPELINE ====================
def run_chrom(chrom, cfg):
    print("\n" + "#"*80)
    print(f"# CHROM {chrom}")
    print("#"*80)

    pc1 = np.load(cfg['pc1_cache'])
    n = len(pc1)
    print(f"pc1: n={n}")

    genes = load_genes(cfg['chrom_plain'])
    genes = genes[(genes['bin'] >= 0) & (genes['bin'] < n)]
    eqtl = load_eqtl()
    merged = genes.merge(eqtl, on='gene_id', how='inner')
    merged = merged.dropna(subset=['neg_log10_p'])
    print(f"genes with eQTL: {len(merged)}")
    if len(merged) < 50:
        print(f"  SKIP chrom {chrom}: only {len(merged)} genes")
        return None

    # real P from cache + self-loop (same rule for real & controls)
    print("Loading real P from cache ...")
    P_real_raw = sp.load_npz(cfg['p_cache'])
    P_real, n_zero_real = apply_self_loop(P_real_raw)
    print(f"  real P: nnz={P_real.nnz}, zero rows patched={n_zero_real}")

    # A_raw from .npy, topk=300
    print(f"Rebuilding A_raw from {os.path.basename(cfg['hic_npy'])} (topk={TOPK}) ...")
    A_raw = rebuild_A_raw(cfg['hic_npy'], TOPK)
    nnz_raw = int((A_raw > 0).sum())
    print(f"  A_raw: nnz={nnz_raw}")

    # controls
    print("Building controls ...")
    t = time.time()
    n_swap_target = N_SWAP_FACTOR * nnz_raw // 2
    A_rew, n_swap = double_edge_swap_numpy(A_raw, n_swap_target, seed=RNG_SEED)
    print(f"  rewired: swaps={n_swap}/{n_swap_target} ({time.time()-t:.1f}s)")
    t = time.time()
    A_shuf = distance_preserving_shuffle(A_raw, seed=RNG_SEED)
    print(f"  shuffled: ({time.time()-t:.1f}s)")
    t = time.time()
    A_flat = flattened_graph(A_raw, seed=RNG_SEED)
    print(f"  flattened: ({time.time()-t:.1f}s)")

    # P for controls
    print("Building P for controls (same residualize + clip + self-loop) ...")
    P_rew, nz_rew = build_P_from_A(sp.csr_matrix(A_rew), pc1)
    print(f"  rewired P: nnz={P_rew.nnz}, zero rows patched={nz_rew}")
    P_shuf, nz_shuf = build_P_from_A(sp.csr_matrix(A_shuf), pc1)
    print(f"  shuffled P: nnz={P_shuf.nnz}, zero rows patched={nz_shuf}")
    P_flat, nz_flat = build_P_from_A(sp.csr_matrix(A_flat), pc1)
    print(f"  flattened P: nnz={P_flat.nnz}, zero rows patched={nz_flat}")

    graphs = {'real': P_real, 'rewired': P_rew, 'shuffled': P_shuf, 'flattened': P_flat}

    # per TF, per graph
    rows = []
    for tf in TFS:
        print(f"\n{'='*80}\nTF: {tf}\n{'='*80}")
        v0 = peaks_to_vector(BED_PATH, chrom, tf, n)
        n_peaks = int(v0.sum())
        print(f"  v0: {n_peaks} bins")
        if n_peaks == 0:
            print(f"  SKIP: no peaks for {tf} on {chrom}")
            continue

        rfa_by_graph = {}
        for gname, P in graphs.items():
            x = rfa_profile(P, v0, GAMMA)
            rfa_by_graph[gname] = x
            print(f"  RFA[{gname}] done")

        bins = merged['bin'].astype(int).values
        y_eqlt = (merged['min_p'] < 5e-8).astype(int).values
        y_cont = merged['neg_log10_p'].values

        if y_eqlt.sum() < 5 or (len(y_eqlt) - y_eqlt.sum()) < 5:
            print(f"  SKIP: too few positives ({int(y_eqlt.sum())}) / negatives")
            continue

        rfa_gene = {g: rfa_by_graph[g][bins] for g in graphs}

        # real
        real = rfa_gene['real']
        auc_real = roc_auc_score(y_eqlt, real)
        auc_neg_real = roc_auc_score(y_eqlt, -real)
        rho_real = spearmanr(real, y_cont).statistic
        ci_auc_real = bootstrap_auc_ci(y_eqlt, real)
        ci_rho_real = bootstrap_rho_ci(real, y_cont)
        print(f"  REAL: AUC(RFA)={auc_real:.3f} CI[{ci_auc_real[0]:.3f},{ci_auc_real[1]:.3f}], "
              f"AUC(-RFA)={auc_neg_real:.3f}, rho={rho_real:+.4f} "
              f"CI[{ci_rho_real[0]:+.4f},{ci_rho_real[1]:+.4f}]")

        rows.append(dict(
            chrom=chrom, graph='real', tf=tf, n_genes=len(merged), n_pos=int(y_eqlt.sum()),
            auc_rfa=auc_real, auc_rfa_lo=ci_auc_real[0], auc_rfa_hi=ci_auc_real[1],
            auc_neg_rfa=auc_neg_real,
            rho_eqlt=rho_real, rho_lo=ci_rho_real[0], rho_hi=ci_rho_real[1],
            delta_auc_vs_real=0.0, delta_auc_lo=np.nan, delta_auc_hi=np.nan,
            delta_rho_vs_real=0.0, delta_rho_lo=np.nan, delta_rho_hi=np.nan,
        ))

        # controls
        for gname in ['rewired', 'shuffled', 'flattened']:
            ctrl = rfa_gene[gname]
            auc_ctrl = roc_auc_score(y_eqlt, ctrl)
            auc_neg_ctrl = roc_auc_score(y_eqlt, -ctrl)
            rho_ctrl = spearmanr(ctrl, y_cont).statistic
            ci_auc_ctrl = bootstrap_auc_ci(y_eqlt, ctrl)
            ci_rho_ctrl = bootstrap_rho_ci(ctrl, y_cont)
            d_auc, d_auc_lo, d_auc_hi = paired_bootstrap_delta(y_eqlt, real, ctrl)
            d_rho, d_rho_lo, d_rho_hi = paired_bootstrap_delta_rho(real, ctrl, y_cont)
            print(f"  {gname:>9}: AUC(RFA)={auc_ctrl:.3f}, AUC(-RFA)={auc_neg_ctrl:.3f}, "
                  f"rho={rho_ctrl:+.4f}, dAUC={d_auc:+.4f} "
                  f"CI[{d_auc_lo:+.4f},{d_auc_hi:+.4f}], "
                  f"drho={d_rho:+.4f} CI[{d_rho_lo:+.4f},{d_rho_hi:+.4f}]")
            rows.append(dict(
                chrom=chrom, graph=gname, tf=tf, n_genes=len(merged), n_pos=int(y_eqlt.sum()),
                auc_rfa=auc_ctrl, auc_rfa_lo=ci_auc_ctrl[0], auc_rfa_hi=ci_auc_ctrl[1],
                auc_neg_rfa=auc_neg_ctrl,
                rho_eqlt=rho_ctrl, rho_lo=ci_rho_ctrl[0], rho_hi=ci_rho_ctrl[1],
                delta_auc_vs_real=d_auc, delta_auc_lo=d_auc_lo, delta_auc_hi=d_auc_hi,
                delta_rho_vs_real=d_rho, delta_rho_lo=d_rho_lo, delta_rho_hi=d_rho_hi,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(cfg['out_csv'], index=False)
    print(f"\nWrote {cfg['out_csv']}")
    return df


# ==================== MAIN ====================
def main():
    t0 = time.time()
    print("="*80)
    print("STEP 3.5: chr17/19 replication of topology controls")
    print(f"N_SWAP_FACTOR={N_SWAP_FACTOR}, TFS={TFS}, gamma={GAMMA}")
    print("="*80)

    all_dfs = []
    for chrom in CHROMS:
        df = run_chrom(chrom, CHROM_CONFIG[chrom])
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo results — aborting summary.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # ========== combined verdict (rho-primary) ==========
    lines = []
    lines.append("="*80)
    lines.append("STEP 3.5 — TOPOLOGY CONTROLS chr17/19, 5 TF, gamma=0.85")
    lines.append("PRIMARY METRIC: rho (delta_rho_vs_real); AUC is supporting")
    lines.append("="*80)
    lines.append(f"N_SWAP_FACTOR={N_SWAP_FACTOR}")
    lines.append("")

    header = (f"{'chrom':<6} {'TF':<8} {'graph':<10} {'AUC_RFA':>8} {'AUC_-RFA':>9} "
              f"{'rho':>8} {'dAUC':>8} {'dAUC_lo':>8} {'dAUC_hi':>8} "
              f"{'drho':>8} {'drho_lo':>8} {'drho_hi':>8}")
    for chrom in CHROMS:
        sub = df[df['chrom'] == chrom]
        if len(sub) == 0: continue
        lines.append("")
        lines.append(f"--- {chrom} ---")
        lines.append(header)
        lines.append("-"*len(header))
        for _, r in sub.iterrows():
            def fmt(v, prec=4, sign=True):
                if pd.isna(v): return "     nan"
                return f"{v:>+8.{prec}f}" if sign else f"{v:>8.{prec}f}"
            lines.append(
                f"{r['chrom']:<6} {r['tf']:<8} {r['graph']:<10} "
                f"{r['auc_rfa']:>8.3f} {r['auc_neg_rfa']:>9.3f} "
                f"{r['rho_eqlt']:>+8.4f} "
                f"{fmt(r['delta_auc_vs_real'])} {fmt(r['delta_auc_lo'])} {fmt(r['delta_auc_hi'])} "
                f"{fmt(r['delta_rho_vs_real'])} {fmt(r['delta_rho_lo'])} {fmt(r['delta_rho_hi'])}"
            )

    # ---- rho-primary significance ----
    ctrl_df = df[df['graph'] != 'real'].copy()
    sig_rho = ((ctrl_df['delta_rho_lo'] > 0) | (ctrl_df['delta_rho_hi'] < 0)) & \
              (ctrl_df['delta_rho_vs_real'].abs() > 0.02)
    sig_auc = ((ctrl_df['delta_auc_lo'] > 0) | (ctrl_df['delta_auc_hi'] < 0)) & \
              (ctrl_df['delta_auc_vs_real'].abs() > 0.02)

    lines.append("")
    lines.append("="*80)
    lines.append("AGGREGATE VERDICT (rho-primary)")
    lines.append("="*80)
    lines.append(f"controls x TF x chrom = {len(ctrl_df)}")
    lines.append(f"PRIMARY   |drho|>0.02 & CI excludes 0: {int(sig_rho.sum())}/{len(ctrl_df)}")
    lines.append(f"secondary |dAUC|>0.02 & CI excludes 0: {int(sig_auc.sum())}/{len(ctrl_df)}")

    # per-control breakdown, per chrom
    lines.append("")
    lines.append("Per-control breakdown (PRIMARY: |drho|>0.02 & CI excludes 0):")
    for gname in ['rewired', 'shuffled', 'flattened']:
        g = ctrl_df[ctrl_df['graph'] == gname]
        if len(g) == 0: continue
        s_all = sig_rho[g.index]
        parts = [f"all {int(s_all.sum())}/{len(g)}"]
        for chrom in CHROMS:
            gg = g[g['chrom'] == chrom]
            if len(gg) == 0: continue
            parts.append(f"{chrom} {int(sig_rho[gg.index].sum())}/{len(gg)}")
        lines.append(f"  {gname:<10}: " + "   ".join(parts))

    # ---- per-chrom verdict ----
    lines.append("")
    lines.append("="*80)
    lines.append("PER-CHROM VERDICT")
    lines.append("="*80)
    for chrom in CHROMS:
        sub = ctrl_df[ctrl_df['chrom'] == chrom]
        if len(sub) == 0: continue
        rew = sub[sub['graph'] == 'rewired']
        shf = sub[sub['graph'] == 'shuffled']
        rew_sig = int(sig_rho[rew.index].sum()) if len(rew) else 0
        shf_sig = int(sig_rho[shf.index].sum()) if len(shf) else 0

        # Direction: d_rho = rho(real) - rho(ctrl).
        # real rho is negative (anticorrelation); if rewired kills it (moves rho
        # toward 0/+), then d_rho < 0 and the CI excludes 0 on the negative side.
        rew = sub[sub['graph'] == 'rewired']
        shf = sub[sub['graph'] == 'shuffled']
        rew_sig = int(sig_rho[rew.index].sum()) if len(rew) else 0
        shf_sig = int(sig_rho[shf.index].sum()) if len(shf) else 0
        rew_rho = rew['rho_eqlt'].mean() if len(rew) else float('nan')
        shf_rho = shf['rho_eqlt'].mean() if len(shf) else float('nan')
        real_rho = df.loc[(df['chrom'] == chrom) & (df['graph'] == 'real'), 'rho_eqlt'].mean()

        lines.append("")
        lines.append(f"--- {chrom} ---")
        lines.append(f"  mean rho: real={real_rho:+.4f}  rewired={rew_rho:+.4f}  shuffled={shf_rho:+.4f}")
        lines.append(f"  shift vs real: rewired={rew_rho - real_rho:+.4f}  shuffled={shf_rho - real_rho:+.4f}")
        lines.append(f"  rewired significant (rho): {rew_sig}/{len(rew)}")
        lines.append(f"  shuffled significant (rho): {shf_sig}/{len(shf)}")

        if rew_sig > 0 and shf_sig == 0:
            lines.append(f"  >>> {chrom}: REPLICATES chr1 PATTERN (rewired kills anticorrelation by rho, shuffled preserves).")
        elif rew_sig == 0 and shf_sig > 0:
            lines.append(f"  >>> {chrom}: INVERTED pattern — shuffled kills anticorrelation by rho,")
            lines.append(f"      rewired preserves it. Consistent with local within-diagonal")
            lines.append(f"      heterogeneity as the carrier on short gene-dense chromosomes.")
        else:
            lines.append(
                f"  >>> {chrom}: rewired does NOT kill anticorrelation by rho — underpowered or structure-specific.")

    # ---- overall conclusion ----
    lines.append("")
    lines.append("="*80)
    lines.append("OVERALL CONCLUSION")
    lines.append("="*80)
    # real anticorrelation check (should hold on both chroms)
    real_df = df[df['graph'] == 'real']
    for chrom in CHROMS:
        r = real_df[real_df['chrom'] == chrom]
        if len(r) == 0: continue
        lines.append(f"  {chrom}: real AUC(RFA) mean={r['auc_rfa'].mean():.3f}, "
                     f"rho mean={r['rho_eqlt'].mean():+.4f}  "
                     f"({int((r['auc_rfa']<0.5).sum())}/{len(r)} TF with AUC<0.5)")
    lines.append("")
    lines.append("  Note: AUC-based delta statistics are underpowered on chr17/19")
    lines.append("  (n_pos~30-50; MDR(dAUC)~0.10-0.15 vs observed ~0.05-0.07).")
    lines.append("  rho is the primary metric on these chromosomes.")

    text = "\n".join(lines)
    print("\n" + text)

    out_txt = os.path.join(OUT_DIR, "topology_controls_chr17_19_5tf.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWrote {out_txt}")

if __name__ == '__main__':
    main()