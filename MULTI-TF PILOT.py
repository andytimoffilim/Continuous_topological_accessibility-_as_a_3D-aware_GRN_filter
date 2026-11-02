#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MULTI-TF PILOT: run RFA pipeline for N TFs on GM12878 chr1 10kb.
Reuses Markov matrix P (computed once, cached).
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu, svds
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
MATRIX_PATH   = r"E:\HIC\results\GM12878_chr1_10kb.npy"
BED_PATH      = r"E:\genomes\tfbs_GM12878.bed"
MATRIX_CSV    = r"E:\genomes\atac_bigwig_results\hic_enhancer_analysis\matrix_with_hic_enhancers.csv"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"
OUT_DIR       = r"E:\HIC\results\pilot_multitf"

RESOLUTION    = 10_000
CHROM         = "chr1"
CHROM_PLAIN   = "1"
CELL_LINE_BED = "GM12878"

GAMMAS        = [0.5, 0.7, 0.85, 0.95]
PR_ALPHA      = 0.85
N_PERM        = 500      # reduced from 1000 for speed
TOPK          = 300

# Diverse TF set: architectural, B-cell, ubiquitous, coactivators, cell cycle
TFS_TO_TEST = [
    'IKZF1', 'CTCF', 'SMC3', 'RAD21', 'PAX5', 'EBF1',
    'EP300', 'POLR2A', 'YY1', 'GABPA', 'MAX', 'E2F4',
    'ELF1', 'TCF12', 'BATF', 'BCL11A', 'IRF4', 'MEF2A',
    'NRF1', 'ZNF143',
]

TF_CATEGORIES = {
    'IKZF1': 'B-cell', 'PAX5': 'B-cell', 'EBF1': 'B-cell',
    'TCF12': 'B-cell', 'BATF': 'B-cell', 'BCL11A': 'B-cell', 'IRF4': 'B-cell',
    'CTCF': 'architectural', 'SMC3': 'architectural', 'RAD21': 'architectural',
    'EP300': 'coactivator', 'POLR2A': 'RNA Pol II',
    'YY1': 'ubiquitous', 'GABPA': 'ubiquitous', 'MAX': 'ubiquitous',
    'E2F4': 'cell cycle', 'ELF1': 'ubiquitous', 'NRF1': 'ubiquitous',
    'ZNF143': 'ubiquitous', 'MEF2A': 'muscle/neuronal',
}


# ==================== MATRIX BUILD ====================

def topk_per_row_sparse(A_dense, k=300):
    n = A_dense.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        row = A_dense[i, :]
        nz = np.nonzero(row)[0]
        if len(nz) == 0: continue
        if len(nz) > k:
            nz = nz[np.argpartition(row[nz], -k)[-k:]]
        rows.extend([i]*len(nz)); cols.extend(nz.tolist()); vals.extend(row[nz].tolist())
    A_sp = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A_sp.maximum(A_sp.T)


def residualize_on_pc1_sparse(A_sp, pc1):
    A_sp = A_sp.tocsr()
    n = A_sp.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        s, e = A_sp.indptr[i], A_sp.indptr[i+1]
        if s == e: continue
        ci, rv = A_sp.indices[s:e], A_sp.data[s:e]
        if len(ci) < 5:
            rows.extend([i]*len(ci)); cols.extend(ci.tolist()); vals.extend(rv.tolist()); continue
        X = np.column_stack([np.ones(len(ci)), pc1[ci]])
        beta, *_ = np.linalg.lstsq(X, rv, rcond=None)
        resid = rv - X @ beta
        keep = resid > 0
        rows.extend([i]*keep.sum())
        cols.extend(ci[keep].tolist()); vals.extend(resid[keep].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_markov_matrix(C, topk=300, residualize_pc1=True):
    n = C.shape[0]
    A = C.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0); A[A < 0] = 0.0
    print(f"    nnz/n^2 initial: {(A > 0).sum()/(n*n):.4f}")

    A_sp = topk_per_row_sparse(A, k=topk)
    print(f"    nnz/n^2 after top-k={topk}: {A_sp.nnz/(n*n):.4f}")

    pc1 = None
    if residualize_pc1:
        print("    PC1 via svds ...")
        C_sp = sp.csr_matrix(C)
        U, S, Vt = svds(C_sp.log1p(), k=1)
        pc1 = U[:, 0] * S[0]
        print("    residualizing on PC1 ...")
        A_sp = residualize_on_pc1_sparse(A_sp, pc1)
        print(f"    nnz/n^2 after residualization: {A_sp.nnz/(n*n):.4f}")

    row_sums = np.array(A_sp.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = (D_inv @ A_sp).tocsr()
    return P, A_sp, pc1


# ==================== RFA / PAGERANK ====================

def rfa_profile(P, v0, gamma, verbose=False):
    n = P.shape[0]
    A_sys = (sp.eye(n, format='csr') - gamma * P).T.tocsc()
    v0 = np.asarray(v0, dtype=np.float64).ravel()
    try:
        if verbose: print(f"    g={gamma:.2f} ILU+gmres ...", end='', flush=True)
        ilu = spilu(A_sys, drop_tol=1e-5, fill_factor=20)
        M = LinearOperator(A_sys.shape, ilu.solve)
        x, info = gmres(A_sys, v0, M=M, rtol=1e-7, atol=0, maxiter=1000)
        if info != 0:
            x, info = gmres(A_sys, v0, M=M, rtol=1e-6, atol=0, maxiter=2000)
        if verbose: print(" done")
        return x
    except Exception as e:
        if verbose: print(f" fallback spsolve ({e})")
        return spsolve(A_sys, v0)


def personalized_pagerank(P, u, alpha=0.85, tol=1e-10, max_iter=2000):
    n = P.shape[0]
    u = u / (u.sum() + 1e-12)
    pr = u.copy(); Pt = P.T.tocsr()
    for it in range(max_iter):
        pr_new = alpha * Pt.dot(pr) + (1 - alpha) * u
        if np.linalg.norm(pr_new - pr, 1) < tol:
            return pr_new, it
        pr = pr_new
    return pr, max_iter


# ==================== BED / GENES / BASELINES ====================

def peaks_to_vector(bed_path, chrom, tf_name, cell_line, resolution, n_bins):
    vec = np.zeros(n_bins, dtype=np.float64); n_peaks = 0
    if not os.path.exists(bed_path): return vec, 0
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6: continue
            if p[0] != chrom or p[3] != tf_name: continue
            if cell_line not in p[5]: continue
            b = int(p[1]) // resolution
            if 0 <= b < n_bins:
                vec[b] = 1.0; n_peaks += 1
    return vec, n_peaks


def load_peaks_bp(bed_path, chrom, tf_name, cell_line):
    out = []
    if not os.path.exists(bed_path): return out
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6: continue
            if p[0] != chrom or p[3] != tf_name: continue
            if cell_line not in p[5]: continue
            out.append((int(p[1]) + int(p[2])) // 2)
    return out


def nearest_peak_distance(gene_start_bp, peak_bp_list):
    if len(peak_bp_list) == 0: return np.inf
    arr = np.array(sorted(peak_bp_list))
    idx = np.searchsorted(arr, gene_start_bp)
    d = np.inf
    if idx < len(arr): d = min(d, abs(arr[idx] - gene_start_bp))
    if idx > 0: d = min(d, abs(arr[idx-1] - gene_start_bp))
    return d


def load_gene_promoters(path, chrom_plain):
    df = pd.read_csv(path, sep=',', low_memory=False)
    df = df[(df['element_type'] == 'promoter') & (df['chrom'].astype(str) == chrom_plain)]
    df = df.sort_values(['gene_id', 'start']).drop_duplicates('gene_id', keep='first')
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    return df[['gene_id', 'start', 'end', 'gene_name']]


# ==================== STATS ====================

def spearman_with_perm(x, y, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    rho, _ = spearmanr(x, y)
    ya = np.asarray(y)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = spearmanr(x, rng.permutation(ya))[0]
    p_perm = float((np.abs(null) >= abs(rho)).mean())
    return float(rho), p_perm


def bootstrap_spearman_ci(x, y, n_boot=500, seed=42):
    rng = np.random.default_rng(seed); n = len(x); rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2: continue
        rhos.append(spearmanr(x[idx], y[idx])[0])
    if not rhos: return np.nan, np.nan
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def partial_spearman(x, y, z_list):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in z_list])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    return float(spearmanr(ex, ey)[0])


# ==================== MAIN ====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    P_CACHE, A_CACHE, PC1_CACHE = (
        os.path.join(OUT_DIR, "P_matrix.npz"),
        os.path.join(OUT_DIR, "A_matrix.npz"),
        os.path.join(OUT_DIR, "pc1.npy"),
    )

    # 1. Load / build P
    if all(os.path.exists(p) for p in [P_CACHE, A_CACHE, PC1_CACHE]):
        print("Loading cached P, A, pc1 ...")
        P = sp.load_npz(P_CACHE); A = sp.load_npz(A_CACHE); pc1 = np.load(PC1_CACHE)
        n = P.shape[0]
        print(f"  P: {P.shape}, nnz/n² = {P.nnz/(n*n):.4f}")
    else:
        print("Loading Hi-C matrix ...")
        C = np.load(MATRIX_PATH); n = C.shape[0]
        print(f"  shape: {C.shape}")
        print("Building Markov matrix (one-time, ~5-10 min) ...")
        P, A, pc1 = build_markov_matrix(C, topk=TOPK, residualize_pc1=True)
        sp.save_npz(P_CACHE, P); sp.save_npz(A_CACHE, A); np.save(PC1_CACHE, pc1)
        print(f"  cached to {OUT_DIR}")

    # 2. Load genes + features
    print("\nLoading genes and features ...")
    genes = load_gene_promoters(GENE_STRUCT, CHROM_PLAIN)
    genes['bin'] = (genes['start'] // RESOLUTION).astype(int)
    genes = genes[(genes['bin'] >= 0) & (genes['bin'] < n)]
    print(f"  promoters: {len(genes)}")

    feat = pd.read_csv(MATRIX_CSV, low_memory=False)
    feat['gene_id'] = feat['gene_id'].astype(str).str.split('.').str[0]

    need = ['gene_id', 'rna', 'atac']
    for tf in TFS_TO_TEST:
        if tf in feat.columns and tf not in need: need.append(tf)
    need = [c for c in need if c in feat.columns]

    merged = genes.merge(feat[need], on='gene_id', how='inner')
    bins = merged['bin'].astype(int).values
    print(f"  after merge: {len(merged)} genes")

    A_csr = A.tocsr()

    # 3. Loop over TFs
    results = []
    for i, tf in enumerate(TFS_TO_TEST, 1):
        print(f"\n{'='*80}\n[{i}/{len(TFS_TO_TEST)}] TF: {tf}\n{'='*80}")
        t0 = time.time()

        if tf not in feat.columns:
            print(f"  SKIP: column '{tf}' not in matrix"); continue

        v0, n_peaks = peaks_to_vector(BED_PATH, CHROM, tf, CELL_LINE_BED, RESOLUTION, n)
        if v0.sum() == 0:
            print(f"  SKIP: no peaks in BED"); continue
        print(f"  v0: {int(v0.sum())} bins ({n_peaks} raw peaks)")

        rfa_by_gamma = {g: rfa_profile(P, v0, g, verbose=True) for g in GAMMAS}
        pr, n_iter = personalized_pagerank(P, v0.copy(), alpha=PR_ALPHA)
        print(f"  PageRank: {n_iter} iters")

        df = merged.copy()
        for g in GAMMAS: df[f'rfa_{g}'] = rfa_by_gamma[g][bins]
        df['rfa_main'] = df[f'rfa_{GAMMAS[2]}']
        df['pr'] = pr[bins]

        # baselines
        peaks_bp = load_peaks_bp(BED_PATH, CHROM, tf, CELL_LINE_BED)
        df['1d_dist'] = df['start'].apply(lambda s: nearest_peak_distance(s, peaks_bp))

        peak_bins = np.where(v0 > 0)[0]
        A_sub = A_csr[:, peak_bins].tocsr()
        dc_full = np.asarray(A_sub.max(axis=1).todense()).flatten()
        df['direct_contact'] = dc_full[bins]

        # validate
        sub = df.dropna(subset=['rfa_main', tf, '1d_dist', 'direct_contact'])
        if len(sub) < 50:
            print(f"  SKIP: only {len(sub)} genes"); continue
        y = sub[tf].values
        if len(np.unique(y)) < 3:
            print(f"  SKIP: target '{tf}' has only {len(np.unique(y))} distinct values")
            continue
        frac_zero = (y == 0).mean()
        print(f"  target: {len(np.unique(y))} distinct, {frac_zero*100:.1f}% zeros, n={len(sub)}")

        x_rfa = sub['rfa_main'].values
        x_1d = sub['1d_dist'].values
        x_dc = sub['direct_contact'].values

        rho_rfa, p_rfa = spearman_with_perm(x_rfa, y, N_PERM)
        rho_1d, p_1d = spearman_with_perm(x_1d, y, N_PERM)
        rho_dc, p_dc = spearman_with_perm(x_dc, y, N_PERM)

        part_1d_dc = partial_spearman(x_rfa, y, [x_1d, x_dc])
        part_pc1 = partial_spearman(x_rfa, y, [pc1[sub['bin'].astype(int).values]])
        part_dc = partial_spearman(x_rfa, y, [x_dc])
        part_1d = partial_spearman(x_rfa, y, [x_1d])
        ci_lo, ci_hi = bootstrap_spearman_ci(x_rfa, y)

        row = {
            'tf': tf, 'category': TF_CATEGORIES.get(tf, 'other'),
            'n_genes': len(sub), 'n_peak_bins': int(v0.sum()),
            'frac_zero': frac_zero,
            'rho_rfa': rho_rfa, 'p_perm_rfa': p_rfa, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'rho_1d': rho_1d, 'p_perm_1d': p_1d,
            'rho_direct': rho_dc, 'p_perm_direct': p_dc,
            'partial_rfa|dc': part_dc,
            'partial_rfa|1d': part_1d,
            'partial_rfa|(1d,dc)': part_1d_dc,
            'partial_rfa|pc1': part_pc1,
        }
        for g in GAMMAS:
            row[f'rho_g{g}'] = spearmanr(sub[f'rfa_{g}'], y)[0]

        results.append(row)

        print(f"  rho(RFA,{tf})  = {rho_rfa:+.4f}  (p_perm={p_rfa:.2e}, CI [{ci_lo:+.3f},{ci_hi:+.3f}])")
        print(f"  rho(1d,{tf})   = {rho_1d:+.4f}")
        print(f"  rho(dc,{tf})   = {rho_dc:+.4f}")
        print(f"  partial RFA | (1d,dc) = {part_1d_dc:+.4f}")
        print(f"  partial RFA | pc1     = {part_pc1:+.4f}")
        print(f"  time: {time.time()-t0:.1f}s")

        pd.DataFrame(results).to_csv(
            os.path.join(OUT_DIR, "pilot_multitf_results.csv"), index=False)

    # 4. Summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "pilot_multitf_results.csv"), index=False)

    lines = [
        "=" * 80,
        f"MULTI-TF PILOT — {len(results)} TFs",
        f"GM12878 chr1 10kb | top-k={TOPK} | gammas={GAMMAS}",
        "=" * 80, "",
        f"{'TF':<10} {'cat':<14} {'N':<5} {'rho_RFA':>9} {'rho_DC':>9} "
        f"{'part|(1d,dc)':>14} {'part|pc1':>10}",
        "-" * 80,
    ]
    for _, r in res_df.sort_values('partial_rfa|(1d,dc)', ascending=False).iterrows():
        lines.append(
            f"{r['tf']:<10} {r['category']:<14} {int(r['n_genes']):<5} "
            f"{r['rho_rfa']:>+9.3f} {r['rho_direct']:>+9.3f} "
            f"{r['partial_rfa|(1d,dc)']:>+14.3f} {r['partial_rfa|pc1']:>+10.3f}"
        )

    n_pos = int((res_df['partial_rfa|(1d,dc)'] > 0.1).sum())
    n_weak = int(((res_df['partial_rfa|(1d,dc)'] >= 0) &
                  (res_df['partial_rfa|(1d,dc)'] <= 0.1)).sum())
    n_neg = int((res_df['partial_rfa|(1d,dc)'] < 0).sum())

    lines += ["", "=" * 80, "AGGREGATE", "=" * 80,
              f"partial RFA|(1d,dc) > 0.1 (strong): {n_pos}/{len(results)}",
              f"partial RFA|(1d,dc) in [0,0.1] (weak): {n_weak}/{len(results)}",
              f"partial RFA|(1d,dc) < 0 (negative): {n_neg}/{len(results)}", ""]
    if n_pos >= 0.6 * len(results):
        lines.append(">>> MAJORITY POSITIVE: RFA is a general 3D-aware GRN filter (publishable).")
    elif n_pos >= 0.3 * len(results):
        lines.append(">>> MIXED: RFA is TF-specific — check categories.")
    else:
        lines.append(">>> MAJORITY NEGATIVE: RFA is IKZF1-specific (negative result).")

    lines += ["", "Breakdown by category (mean partial |(1d,dc)):"]
    for cat, grp in res_df.groupby('category'):
        lines.append(f"  {cat:<16}: {grp['partial_rfa|(1d,dc)'].mean():+.3f} (n={len(grp)})")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "pilot_multitf_summary.txt"), 'w') as f:
        f.write(text)

    print(f"\nSaved:\n  {os.path.join(OUT_DIR, 'pilot_multitf_results.csv')}\n"
          f"  {os.path.join(OUT_DIR, 'pilot_multitf_summary.txt')}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()