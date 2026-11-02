#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eQTL multi-chromosome test (chr1, chr17, chr19) for GM12878 10kb.
Builds P per chromosome (cached), computes RFA/1D/dc per TF per chromosome,
then aggregates across chromosomes and reports AUC and partial correlations.

Target: eqtl_bin (1 if gene has >=1 variant with p<5e-8 in GTEx v10 LCL).
Also continuous target: -log10(min_p).

Outputs: eqtl_multi_chr_summary.txt, eqtl_multi_chr_results.csv,
         P_chr{N}.npz, A_chr{N}.npz, pc1_chr{N}.npy (cached per chromosome).
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu, svds
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
HIC_DIR       = r"E:\HIC\results"
P_CACHE_CHR1  = r"E:\HIC\results\pilot_multitf\P_matrix.npz"
A_CACHE_CHR1  = r"E:\HIC\results\pilot_multitf\A_matrix.npz"
PC1_CACHE_CHR1= r"E:\HIC\results\pilot_multitf\pc1.npy"

MATRIX_CSV    = r"E:\genomes\atac_bigwig_results\hic_enhancer_analysis\matrix_with_hic_enhancers.csv"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"
BED_PATH      = r"E:\genomes\tfbs_GM12878.bed"
EQTL_CSV      = r"E:\HIC\eqtl\eqtl_target.csv"
OUT_DIR       = r"E:\HIC\results\eqtl_multi_chr"

RESOLUTION    = 10_000
CELL_LINE     = "GM12878"
CHROMS        = [1, 17, 19]
TOPK          = 300
GAMMA_MAIN    = 0.85
N_PERM        = 500
N_BOOT        = 1000
RNG_SEED      = 42

TFS_TO_TEST = [
    'IKZF1', 'CTCF', 'SMC3', 'RAD21', 'PAX5', 'EBF1',
    'EP300', 'POLR2A', 'YY1', 'GABPA', 'MAX', 'E2F4',
    'ELF1', 'TCF12', 'BATF', 'BCL11A', 'IRF4', 'MEF2A',
    'NRF1', 'ZNF143',
]


# ==================== MATRIX BUILD ====================

def topk_per_row_sparse(A_dense, k=TOPK):
    n = A_dense.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        row = A_dense[i, :]
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            continue
        if len(nz) > k:
            nz = nz[np.argpartition(row[nz], -k)[-k:]]
        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(row[nz].tolist())
    A_sp = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A_sp.maximum(A_sp.T)


def residualize_on_pc1_sparse(A_sp, pc1):
    A_sp = A_sp.tocsr()
    n = A_sp.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        s, e = A_sp.indptr[i], A_sp.indptr[i + 1]
        if s == e:
            continue
        ci, rv = A_sp.indices[s:e], A_sp.data[s:e]
        if len(ci) < 5:
            rows.extend([i] * len(ci))
            cols.extend(ci.tolist())
            vals.extend(rv.tolist())
            continue
        X = np.column_stack([np.ones(len(ci)), pc1[ci]])
        beta, *_ = np.linalg.lstsq(X, rv, rcond=None)
        resid = rv - X @ beta
        keep = resid > 0
        rows.extend([i] * keep.sum())
        cols.extend(ci[keep].tolist())
        vals.extend(resid[keep].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_markov_matrix(C, topk=TOPK, residualize_pc1=True):
    n = C.shape[0]
    A = C.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    A[A < 0] = 0.0
    A_sp = topk_per_row_sparse(A, k=topk)
    pc1 = None
    if residualize_pc1:
        C_sp = sp.csr_matrix(C)
        U, S, Vt = svds(C_sp.log1p(), k=1)
        pc1 = U[:, 0] * S[0]
        A_sp = residualize_on_pc1_sparse(A_sp, pc1)
    row_sums = np.array(A_sp.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = (D_inv @ A_sp).tocsr()
    return P, A_sp, pc1


def load_or_build_P(chrom):
    if chrom == 1 and all(os.path.exists(p) for p in [P_CACHE_CHR1, A_CACHE_CHR1, PC1_CACHE_CHR1]):
        print(f"  chr{chrom}: loading cached P from {P_CACHE_CHR1}")
        return sp.load_npz(P_CACHE_CHR1), sp.load_npz(A_CACHE_CHR1), np.load(PC1_CACHE_CHR1)

    npy = os.path.join(HIC_DIR, f"GM12878_chr{chrom}_10kb.npy")
    P_cache = os.path.join(OUT_DIR, f"P_chr{chrom}.npz")
    A_cache = os.path.join(OUT_DIR, f"A_chr{chrom}.npz")
    pc1_cache = os.path.join(OUT_DIR, f"pc1_chr{chrom}.npy")

    if all(os.path.exists(p) for p in [P_cache, A_cache, pc1_cache]):
        print(f"  chr{chrom}: loading cached P from {P_cache}")
        return sp.load_npz(P_cache), sp.load_npz(A_cache), np.load(pc1_cache)

    print(f"  chr{chrom}: building P from {npy} ...")
    C = np.load(npy)
    print(f"    C: {C.shape}, nnz/n² = {(C > 0).mean():.4f}")
    P, A, pc1 = build_markov_matrix(C, topk=TOPK, residualize_pc1=True)
    sp.save_npz(P_cache, P)
    sp.save_npz(A_cache, A)
    np.save(pc1_cache, pc1)
    print(f"    cached to {OUT_DIR}")
    return P, A, pc1


# ==================== LOADERS ====================

def load_gene_promoters(path, chrom_plain):
    df = pd.read_csv(path, sep=',', low_memory=False)
    df = df[(df['element_type'] == 'promoter') &
            (df['chrom'].astype(str) == str(chrom_plain))]
    df = df.sort_values(['gene_id', 'start']).drop_duplicates('gene_id', keep='first')
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    return df[['gene_id', 'start', 'end', 'gene_name']]


def load_eqtl(path):
    df = pd.read_csv(path)
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    df = df.drop_duplicates('gene_id', keep='first')
    return df[['gene_id', 'min_p', 'n_sig', 'eqtl_bin']]


def peaks_to_vector(bed_path, chrom, tf, cell, resolution, n_bins):
    vec = np.zeros(n_bins)
    if not os.path.exists(bed_path):
        return vec
    chrom_str = f"chr{chrom}"
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6:
                continue
            if p[0] != chrom_str or p[3] != tf:
                continue
            if cell not in p[5]:
                continue
            b = int(p[1]) // resolution
            if 0 <= b < n_bins:
                vec[b] = 1.0
    return vec


def load_peaks_bp(bed_path, chrom, tf, cell):
    out = []
    if not os.path.exists(bed_path):
        return out
    chrom_str = f"chr{chrom}"
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6:
                continue
            if p[0] != chrom_str or p[3] != tf:
                continue
            if cell not in p[5]:
                continue
            out.append((int(p[1]) + int(p[2])) // 2)
    return out


def nearest_peak_distance(gene_start_bp, peak_bp_list):
    if len(peak_bp_list) == 0:
        return np.inf
    arr = np.array(sorted(peak_bp_list))
    idx = np.searchsorted(arr, gene_start_bp)
    d = np.inf
    if idx < len(arr):
        d = min(d, abs(arr[idx] - gene_start_bp))
    if idx > 0:
        d = min(d, abs(arr[idx - 1] - gene_start_bp))
    return d


# ==================== RFA ====================

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


# ==================== STATS ====================

def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_auc_diff(y_true, y_a, y_b, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_obs = roc_auc_score(y_true, y_a) - roc_auc_score(y_true, y_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            d = roc_auc_score(yt, y_a[idx]) - roc_auc_score(yt, y_b[idx])
            deltas.append(d)
        except ValueError:
            continue
    if not deltas:
        return delta_obs, np.nan, np.nan, np.nan
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_one = float((deltas <= 0).mean())
    return float(delta_obs), float(ci_lo), float(ci_hi), p_one


def spearman_with_perm(x, y, n_perm=N_PERM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rho, _ = spearmanr(x, y)
    if not np.isfinite(rho):
        return np.nan, np.nan
    ya = np.asarray(y)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = spearmanr(x, rng.permutation(ya))[0]
    return float(rho), float((np.abs(null) >= abs(rho)).mean())


def partial_spearman(x, y, z_list):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in z_list])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    return float(spearmanr(ex, ey)[0])


def bootstrap_partial_ci(x, y, z_list, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = partial_spearman(x[idx], y[idx], [z[idx] for z in z_list])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ==================== MAIN ====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print("eQTL multi-chromosome test (chr1, chr17, chr19)")
    print("=" * 80)

    eqtl = load_eqtl(EQTL_CSV)
    print(f"  eQTL genes total: {len(eqtl)}, with sig variant: {int(eqtl['eqtl_bin'].sum())}")

    feat = pd.read_csv(MATRIX_CSV, low_memory=False)
    feat['gene_id'] = feat['gene_id'].astype(str).str.split('.').str[0]
    need = ['gene_id'] + [tf for tf in TFS_TO_TEST if tf in feat.columns]

    per_chrom_data = {}  # chrom -> dict(tf -> DataFrame with rfa, 1d, dc, y)

    for chrom in CHROMS:
        print(f"\n{'=' * 80}\nChromosome {chrom}\n{'=' * 80}")
        P, A, pc1 = load_or_build_P(chrom)
        n = P.shape[0]
        print(f"  P: {P.shape}, nnz/n² = {P.nnz / (n * n):.4f}")

        genes = load_gene_promoters(GENE_STRUCT, chrom)
        genes['bin'] = (genes['start'] // RESOLUTION).astype(int)
        genes = genes[(genes['bin'] >= 0) & (genes['bin'] < n)]
        print(f"  promoters on chr{chrom}: {len(genes)}")

        if chrom == 1:
            # chr1: keep original filter through feat (811 genes, совместимо с прежними тестами)
            merged_base = genes.merge(feat[need], on='gene_id', how='inner')
        else:
            # chr17/19: feat is chr1-only, so use all promoters on this chromosome
            merged_base = genes.copy()
            for tf in TFS_TO_TEST:
                if tf not in merged_base.columns:
                    merged_base[tf] = 0  # placeholder, not used downstream
        merged_base = merged_base.merge(eqtl, on='gene_id', how='inner')


        print(f"  after merge with features & eQTL: {len(merged_base)}")
        print(f"  eqtl_bin=1: {int(merged_base['eqtl_bin'].sum())}, "
              f"eqtl_bin=0: {int((merged_base['eqtl_bin'] == 0).sum())}")

        A_csr = A.tocsr()
        per_chrom_data[chrom] = {}

        for tf in TFS_TO_TEST:
            if tf not in merged_base.columns:
                continue
            v0 = peaks_to_vector(BED_PATH, chrom, tf, CELL_LINE, RESOLUTION, n)
            if v0.sum() == 0:
                continue
            x_rfa = rfa_profile(P, v0, GAMMA_MAIN)

            peak_bins = np.where(v0 > 0)[0]
            A_sub = A_csr[:, peak_bins].tocsr()
            dc_full = np.asarray(A_sub.max(axis=1).todense()).flatten()

            merged = merged_base.copy()
            merged['rfa'] = x_rfa[merged['bin'].astype(int).values]
            merged['dc'] = dc_full[merged['bin'].astype(int).values]

            peaks_bp = load_peaks_bp(BED_PATH, chrom, tf, CELL_LINE)
            merged['1d'] = merged['start'].apply(
                lambda s: nearest_peak_distance(s, peaks_bp))

            valid = (~np.isnan(merged['rfa']) & ~np.isnan(merged['dc']) &
                     np.isfinite(merged['1d']) & ~np.isnan(merged['eqtl_bin']))
            sub = merged[valid]
            if len(sub) < 20:
                continue
            per_chrom_data[chrom][tf] = sub[['gene_id', 'rfa', '1d', 'dc',
                                              'eqtl_bin', 'min_p']].copy()

    # ==================== AGGREGATE PER TF ====================
    all_results = []
    for tf in TFS_TO_TEST:
        parts = [per_chrom_data[c][tf] for c in CHROMS if tf in per_chrom_data.get(c, {})]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        n_valid = len(df)
        y_bin = df['eqtl_bin'].values.astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(n_valid - n_pos)
        if n_pos < 10 or n_neg < 10:
            print(f"  {tf}: SKIP (pos={n_pos}, neg={n_neg})")
            continue

        rfa = df['rfa'].values
        dc = df['dc'].values
        x_1d = df['1d'].values

        auc_rfa = float(roc_auc_score(y_bin, rfa))
        auc_rfa_neg = float(roc_auc_score(y_bin, -rfa))
        rfa_for_auc = -rfa if auc_rfa_neg > auc_rfa else rfa
        auc_rfa = max(auc_rfa, auc_rfa_neg)

        auc_1d = float(roc_auc_score(y_bin, -x_1d))
        auc_dc = float(roc_auc_score(y_bin, dc))
        z_rfa = (rfa_for_auc - rfa_for_auc.mean()) / (rfa_for_auc.std() + 1e-10)
        z_dc = (dc - dc.mean()) / (dc.std() + 1e-10)
        auc_comb = float(roc_auc_score(y_bin, z_rfa + z_dc))

        delta_rfa_1d, d_lo, d_hi, p_delta = bootstrap_auc_diff(
            y_bin, rfa_for_auc, -x_1d, N_BOOT)

        y_cont = -np.log10(np.clip(df['min_p'].values, 1e-300, 1.0))
        rho_rfa, p_rfa = spearman_with_perm(rfa, y_cont)
        rho_1d, p_1d = spearman_with_perm(-x_1d, y_cont)
        rho_dc, p_dc = spearman_with_perm(dc, y_cont)
        part_1d_dc = partial_spearman(rfa, y_cont, [x_1d, dc])
        part_lo, part_hi = bootstrap_partial_ci(rfa, y_cont, [x_1d, dc])

        row = {
            'tf': tf, 'n_genes': n_valid, 'n_pos': n_pos, 'n_neg': n_neg,
            'auc_rfa': auc_rfa, 'auc_1d': auc_1d, 'auc_dc': auc_dc,
            'auc_comb': auc_comb,
            'delta_auc_rfa_minus_1d': delta_rfa_1d,
            'delta_lo': d_lo, 'delta_hi': d_hi, 'p_boot': p_delta,
            'rho_rfa_cont': rho_rfa, 'p_perm_rfa': p_rfa,
            'rho_1d_cont': rho_1d, 'rho_dc_cont': rho_dc,
            'partial_rfa_1d_dc': part_1d_dc,
            'partial_lo': part_lo, 'partial_hi': part_hi,
        }
        for c in CHROMS:
            if tf in per_chrom_data.get(c, {}):
                row[f'n_chr{c}'] = len(per_chrom_data[c][tf])
            else:
                row[f'n_chr{c}'] = 0
        all_results.append(row)
        pd.DataFrame(all_results).to_csv(
            os.path.join(OUT_DIR, "eqtl_multi_chr_results.csv"), index=False)

    res = pd.DataFrame(all_results)
    res.to_csv(os.path.join(OUT_DIR, "eqtl_multi_chr_results.csv"), index=False)

    lines = [
        "=" * 160,
        f"eQTL multi-chr target (GTEx v10 LCL, chr1+17+19) — {len(res)} TFs",
        f"GM12878 10kb | gamma={GAMMA_MAIN} | N_boot={N_BOOT} | topk={TOPK}",
        "=" * 160, "",
        f"{'TF':<10} {'N':<5} {'pos':<5} {'neg':<5} "
        f"{'AUC_RFA':>8} {'AUC_1D':>8} {'AUC_dc':>8} {'AUC_comb':>9} "
        f"{'dAUC(RFA−1D)':>13} {'CI_lo':>8} {'CI_hi':>8} {'p_boot':>7} "
        f"{'rho_RFA':>8} {'rho_1D':>8} {'part|(1D,dc)':>13} {'part_lo':>8} {'part_hi':>8}",
        "-" * 160,
    ]
    for _, r in res.sort_values('delta_auc_rfa_minus_1d', ascending=False).iterrows():
        lines.append(
            f"{r['tf']:<10} {int(r['n_genes']):<5} {int(r['n_pos']):<5} "
            f"{int(r['n_neg']):<5} "
            f"{r['auc_rfa']:>8.3f} {r['auc_1d']:>8.3f} {r['auc_dc']:>8.3f} "
            f"{r['auc_comb']:>9.3f} "
            f"{r['delta_auc_rfa_minus_1d']:>+13.4f} "
            f"{r['delta_lo']:>+8.3f} {r['delta_hi']:>+8.3f} {r['p_boot']:>7.3f} "
            f"{r['rho_rfa_cont']:>+8.3f} {r['rho_1d_cont']:>+8.3f} "
            f"{r['partial_rfa_1d_dc']:>+13.4f} "
            f"{r['partial_lo']:>+8.3f} {r['partial_hi']:>+8.3f}"
        )

    n_rfa_better = int((res['delta_auc_rfa_minus_1d'] > 0).sum())
    n_ci_pos = int((res['delta_lo'] > 0).sum())
    n_part_pos = int((res['partial_rfa_1d_dc'] > 0).sum())
    n_part_ci_pos = int((res['partial_lo'] > 0).sum())

    lines += [
        "", "=" * 160, "AGGREGATE", "=" * 160,
        f"Mean AUC RFA  : {res['auc_rfa'].mean():.3f}",
        f"Mean AUC 1D   : {res['auc_1d'].mean():.3f}",
        f"Mean AUC dc   : {res['auc_dc'].mean():.3f}",
        f"Mean AUC comb : {res['auc_comb'].mean():.3f}",
        "",
        f"RFA > 1D (any)   : {n_rfa_better}/{len(res)}",
        f"RFA > 1D (CI>0)  : {n_ci_pos}/{len(res)}",
        f"partial RFA|(1D,dc) > 0  : {n_part_pos}/{len(res)}",
        f"partial RFA|(1D,dc) CI>0 : {n_part_ci_pos}/{len(res)}  <- main functional claim",
        "",
    ]

    if n_part_ci_pos >= 0.6 * len(res):
        lines.append(">>> RFA ADDS over 1D+dc on eQTL target for MAJORITY. "
                     "Functional claim supported.")
    elif n_part_ci_pos >= 0.3 * len(res):
        lines.append(">>> MIXED: RFA adds over 1D+dc for some TFs.")
    else:
        lines.append(">>> RFA does NOT add over 1D+dc on eQTL target. "
                     "Check power (few positives) or layer mismatch.")

    if n_rfa_better >= 0.5 * len(res):
        lines.append(">>> RFA beats 1D by AUC for majority of TFs.")
    else:
        lines.append(">>> 1D still beats RFA by AUC — check power and sign flip.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "eqtl_multi_chr_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\nSaved:\n  {os.path.join(OUT_DIR, 'eqtl_multi_chr_results.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'eqtl_multi_chr_summary.txt')}")


if __name__ == '__main__':
    main()