#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Delta-RFA vs Delta-RNA (MCF10A vs MCF7, 40kb, chr1/17/19).

v0 = ATAC peaks per cell line (not TF peaks) -> RFA per gene -> ΔRFA.
Target = ΔRNA = log2(TPM+1)_MCF7 - log2(TPM+1)_MCF10A.
Tests: Spearman(ΔRFA, ΔRNA), partial | baseline RNA, bootstrap CI.
Also sanity: rho(RFA_line, RNA_line) within each line.

Outputs: delta_rfa_summary.txt, delta_rfa_results.csv,
         P_MCF{10A,7}_chr{N}.npz, A_..., pc1_... (cached).
"""

import os
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu, svds
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
HIC_DIR       = r"E:\HIC\results"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"
RNA_MCF10A    = r"E:\HIC\rnaseq\MCF10A_RNA.tsv"
RNA_MCF7      = r"E:\HIC\rnaseq\MCF7_RNA.tsv"
ATAC_MCF7     = r"E:\HIC\results\GSE169929_ENCFF882OVP_IDR_thresholded_peaks_GRCh38.bed.gz"
ATAC_MCF10A   = r"E:\HIC\results\GSM3735772_MCF10A_ATAC_peaks.broadPeak.gz"
OUT_DIR       = r"E:\HIC\results\delta_rfa"

RESOLUTION    = 40_000
CHROMS        = [1, 17, 19]
TOPK          = 300
GAMMA_MAIN    = 0.85
N_PERM        = 500
N_BOOT        = 1000
RNG_SEED      = 42


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


def load_or_build_P(line, chrom):
    npy = os.path.join(HIC_DIR, f"{line}_chr{chrom}_40kb_fixed.npy")
    P_cache = os.path.join(OUT_DIR, f"P_{line}_chr{chrom}.npz")
    A_cache = os.path.join(OUT_DIR, f"A_{line}_chr{chrom}.npz")
    pc1_cache = os.path.join(OUT_DIR, f"pc1_{line}_chr{chrom}.npy")

    if all(os.path.exists(p) for p in [P_cache, A_cache, pc1_cache]):
        print(f"    {line} chr{chrom}: cached")
        return sp.load_npz(P_cache), sp.load_npz(A_cache), np.load(pc1_cache)

    print(f"    {line} chr{chrom}: building P from {npy} ...")
    C = np.load(npy)
    print(f"      C: {C.shape}, nnz/n² = {(C > 0).mean():.4f}")
    P, A, pc1 = build_markov_matrix(C, topk=TOPK, residualize_pc1=True)
    sp.save_npz(P_cache, P)
    sp.save_npz(A_cache, A)
    np.save(pc1_cache, pc1)
    return P, A, pc1


# ==================== LOADERS ====================

def load_atac_peaks(path, chrom):
    """Return set of bins (start // RESOLUTION) with ATAC peak on chr{chrom}."""
    bins = set()
    chrom_str = f"chr{chrom}"
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 3:
                continue
            if p[0] != chrom_str:
                continue
            try:
                start = int(p[1])
            except ValueError:
                continue
            bins.add(start // RESOLUTION)
    return bins


def load_gene_promoters(path, chrom_plain):
    df = pd.read_csv(path, sep=',', low_memory=False)
    df = df[(df['element_type'] == 'promoter') &
            (df['chrom'].astype(str) == str(chrom_plain))]
    df = df.sort_values(['gene_id', 'start']).drop_duplicates('gene_id', keep='first')
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    return df[['gene_id', 'start', 'end', 'gene_name']]


def load_rna(path):
    df = pd.read_csv(path, sep='\t')
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    df = df.drop_duplicates('gene_id', keep='first')
    return df[['gene_id', 'TPM']]


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


def bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        r = spearmanr(x[idx], y[idx])[0]
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ==================== MAIN ====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print("ΔRFA vs ΔRNA (MCF10A vs MCF7, 40kb, chr1/17/19)")
    print("=" * 80)

    print("\nLoading RNA ...")
    rna_10a = load_rna(RNA_MCF10A).rename(columns={'TPM': 'TPM_10A'})
    rna_7 = load_rna(RNA_MCF7).rename(columns={'TPM': 'TPM_7'})
    print(f"  MCF10A: {len(rna_10a)}, MCF7: {len(rna_7)}")

    rows = []
    for chrom in CHROMS:
        print(f"\n{'=' * 80}\nChromosome {chrom}\n{'=' * 80}")

        genes = load_gene_promoters(GENE_STRUCT, chrom)
        genes['bin'] = (genes['start'] // RESOLUTION).astype(int)
        print(f"  promoters: {len(genes)}")

        rfa_per_line = {}
        for line, atac_path in [('MCF10A', ATAC_MCF10A), ('MCF7', ATAC_MCF7)]:
            print(f"\n  --- {line} ---")
            P, A, pc1 = load_or_build_P(line, chrom)
            n = P.shape[0]
            print(f"    P: {P.shape}, nnz/n² = {P.nnz / (n * n):.4f}")
            genes_line = genes[(genes['bin'] >= 0) & (genes['bin'] < n)].copy()
            print(f"    genes in range: {len(genes_line)}")

            atac_bins = load_atac_peaks(atac_path, chrom)
            atac_bins = {b for b in atac_bins if 0 <= b < n}
            print(f"    ATAC bins: {len(atac_bins)}")
            if len(atac_bins) == 0:
                print(f"    SKIP: no ATAC peaks on chr{chrom}")
                rfa_per_line[line] = None
                continue

            v0 = np.zeros(n)
            for b in atac_bins:
                v0[b] = 1.0
            x_rfa = rfa_profile(P, v0, GAMMA_MAIN)
            rfa_per_line[line] = (genes_line, x_rfa)

        if rfa_per_line.get('MCF10A') is None or rfa_per_line.get('MCF7') is None:
            print(f"  SKIP chr{chrom}: missing ATAC for one line")
            continue

        genes_10a, x_10a = rfa_per_line['MCF10A']
        genes_7, x_7 = rfa_per_line['MCF7']

        # intersect genes by gene_id
        merged = genes_10a[['gene_id', 'bin']].merge(
            genes_7[['gene_id', 'bin']], on='gene_id', suffixes=('_10a', '_7'))
        merged['RFA_10A'] = x_10a[merged['bin_10a'].astype(int).values]
        merged['RFA_7'] = x_7[merged['bin_7'].astype(int).values]
        merged['dRFA'] = merged['RFA_7'] - merged['RFA_10A']

        # RNA
        merged = merged.merge(rna_10a, on='gene_id', how='inner')
        merged = merged.merge(rna_7, on='gene_id', how='inner')
        merged['log10A'] = np.log2(merged['TPM_10A'] + 1)
        merged['log7'] = np.log2(merged['TPM_7'] + 1)
        merged['dRNA'] = merged['log7'] - merged['log10A']

        merged['chrom'] = chrom
        rows.append(merged)

    if not rows:
        print("\nNo data — abort.")
        return

    df = pd.concat(rows, ignore_index=True)
    print(f"\n{'=' * 80}\nCombined: {len(df)} genes across chr{CHROMS}\n{'=' * 80}")

    # drop genes with zero RNA in both lines (not expressed -> dRNA meaningless)
    df = df[(df['TPM_10A'] > 0) | (df['TPM_7'] > 0)].copy()
    print(f"  after dropping unexpressed: {len(df)}")

    x = df['dRFA'].values
    y = df['dRNA'].values
    z1 = df['log10A'].values
    z2 = df['log7'].values

    rho, p_perm = spearman_with_perm(x, y)
    rho_10a, p_10a = spearman_with_perm(df['RFA_10A'].values, df['log10A'].values)
    rho_7, p_7 = spearman_with_perm(df['RFA_7'].values, df['log7'].values)
    part_base = partial_spearman(x, y, [z1, z2])
    ci_lo, ci_hi = bootstrap_spearman_ci(x, y)

    print(f"\n  rho(ΔRFA, ΔRNA)              = {rho:+.4f}  (p_perm={p_perm:.2e}, CI [{ci_lo:+.3f}, {ci_hi:+.3f}])")
    print(f"  rho(RFA_MCF10A, RNA_MCF10A)  = {rho_10a:+.4f}  (p_perm={p_10a:.2e})")
    print(f"  rho(RFA_MCF7,   RNA_MCF7)    = {rho_7:+.4f}  (p_perm={p_7:.2e})")
    print(f"  partial rho(ΔRFA, ΔRNA | baseline RNA) = {part_base:+.4f}")

    summary = [
        "=" * 120,
        "ΔRFA vs ΔRNA (MCF10A vs MCF7, 40kb, chr1/17/19)",
        f"gamma={GAMMA_MAIN} | topk={TOPK} | N_perm={N_PERM} | N_boot={N_BOOT}",
        "=" * 120, "",
        f"Genes total (expressed): {len(df)}",
        f"  chr1:  {int((df['chrom']==1).sum())}",
        f"  chr17: {int((df['chrom']==17).sum())}",
        f"  chr19: {int((df['chrom']==19).sum())}",
        "",
        f"rho(ΔRFA, ΔRNA)             = {rho:+.4f}  (p_perm={p_perm:.2e}, CI [{ci_lo:+.3f}, {ci_hi:+.3f}])",
        f"partial | baseline RNA      = {part_base:+.4f}",
        "",
        "Sanity (within-line):",
        f"  rho(RFA_MCF10A, RNA_MCF10A) = {rho_10a:+.4f}  (p_perm={p_10a:.2e})",
        f"  rho(RFA_MCF7,   RNA_MCF7)   = {rho_7:+.4f}  (p_perm={p_7:.2e})",
        "",
    ]

    if ci_lo > 0:
        summary.append(">>> ΔRFA POSITIVELY predicts ΔRNA (CI_lo > 0). "
                       "Continuous 3D accessibility tracks regulatory change.")
    elif ci_hi < 0:
        summary.append(">>> ΔRFA NEGATIVELY predicts ΔRNA (CI_hi < 0). "
                       "Check sign convention / baseline RNA control.")
    else:
        summary.append(">>> ΔRFA does NOT predict ΔRNA (CI spans 0).")

    if part_base > 0.05:
        summary.append(">>> partial | baseline RNA is positive — ΔRFA adds beyond expression level.")
    elif part_base < -0.05:
        summary.append(">>> partial | baseline RNA is negative — ΔRFA likely proxying baseline expression.")
    else:
        summary.append(">>> partial | baseline RNA is near zero.")

    text = "\n".join(summary)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "delta_rfa_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(text)

    df[['gene_id', 'chrom', 'bin_10a', 'bin_7', 'RFA_10A', 'RFA_7', 'dRFA',
        'TPM_10A', 'TPM_7', 'log10A', 'log7', 'dRNA']].to_csv(
        os.path.join(OUT_DIR, "delta_rfa_results.csv"), index=False)

    print(f"\nSaved:\n  {os.path.join(OUT_DIR, 'delta_rfa_summary.txt')}")
    print(f"  {os.path.join(OUT_DIR, 'delta_rfa_results.csv')}")


if __name__ == '__main__':
    main()