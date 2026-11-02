#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eQTL dose-response: does AUC(RFA) grow with eQTL strength?
Builds RFA per-TF per-chromosome (chr1/17/19) and caches to eqtl_rfa_cache.npz.
Then:
  - AUC(RFA) vs "no eQTL" across 4 quantiles of -log10(min_p)
  - AUC(RFA) vs "no eQTL" across 5 bins of n_sig (0,1,2,3-5,>=6)
  - permutation null for AUC(RFA) on eqtl_bin
  - per-chromosome AUC breakdown
Outputs: eqtl_dose_response.txt, eqtl_dose_response.csv
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
HIC_DIR       = r"E:\HIC\results"
P_CACHE_CHR1  = r"E:\HIC\results\pilot_multitf\P_matrix.npz"
A_CACHE_CHR1  = r"E:\HIC\results\pilot_multitf\A_matrix.npz"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"
BED_PATH      = r"E:\genomes\tfbs_GM12878.bed"
EQTL_CSV      = r"E:\HIC\eqtl\eqtl_target.csv"
OUT_DIR       = r"E:\HIC\results\eqtl_dose"
RFA_CACHE     = os.path.join(OUT_DIR, "eqtl_rfa_cache.npz")

RESOLUTION    = 10_000
CELL_LINE     = "GM12878"
CHROMS        = [1, 17, 19]
GAMMA_MAIN    = 0.85
N_PERM        = 500
RNG_SEED      = 42

TFS_TO_TEST = [
    'IKZF1', 'CTCF', 'SMC3', 'RAD21', 'PAX5', 'EBF1',
    'EP300', 'POLR2A', 'YY1', 'GABPA', 'MAX', 'E2F4',
    'ELF1', 'TCF12', 'BATF', 'BCL11A', 'IRF4', 'MEF2A',
    'NRF1', 'ZNF143',
]


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
    chrom_str = f"chr{chrom}"
    if not os.path.exists(bed_path):
        return vec
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6: continue
            if p[0] != chrom_str or p[3] != tf: continue
            if cell not in p[5]: continue
            b = int(p[1]) // resolution
            if 0 <= b < n_bins:
                vec[b] = 1.0
    return vec


def load_or_build_P(chrom):
    if chrom == 1 and os.path.exists(P_CACHE_CHR1):
        return sp.load_npz(P_CACHE_CHR1), sp.load_npz(A_CACHE_CHR1), None
    P_cache = os.path.join(HIC_DIR, f"eqtl_multi_chr", f"P_chr{chrom}.npz")
    A_cache = os.path.join(HIC_DIR, f"eqtl_multi_chr", f"A_chr{chrom}.npz")
    if os.path.exists(P_cache):
        return sp.load_npz(P_cache), sp.load_npz(A_cache), None
    raise FileNotFoundError(f"P cache missing for chr{chrom}: {P_cache}")


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


def build_rfa_cache():
    """Build RFA per TF per chromosome, save to npz."""
    print("Building RFA cache (first run, ~30-60 min) ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    eqtl = load_eqtl(EQTL_CSV)
    store = {}

    for chrom in CHROMS:
        print(f"\nChromosome {chrom}")
        P, A, _ = load_or_build_P(chrom)
        n = P.shape[0]
        genes = load_gene_promoters(GENE_STRUCT, chrom)
        genes['bin'] = (genes['start'] // RESOLUTION).astype(int)
        genes = genes[(genes['bin'] >= 0) & (genes['bin'] < n)]
        genes = genes.merge(eqtl, on='gene_id', how='inner')
        print(f"  genes after merge with eQTL: {len(genes)}")
        bins = genes['bin'].astype(int).values
        gids = genes['gene_id'].values

        store[f'chr{chrom}_gene_id'] = gids
        store[f'chr{chrom}_bin'] = bins
        store[f'chr{chrom}_eqtl_bin'] = genes['eqtl_bin'].values
        store[f'chr{chrom}_min_p'] = genes['min_p'].values
        store[f'chr{chrom}_n_sig'] = genes['n_sig'].values

        for tf in TFS_TO_TEST:
            v0 = peaks_to_vector(BED_PATH, chrom, tf, CELL_LINE, RESOLUTION, n)
            if v0.sum() == 0:
                print(f"    {tf}: SKIP (no peaks)")
                continue
            x_rfa = rfa_profile(P, v0, GAMMA_MAIN)
            store[f'chr{chrom}_{tf}_rfa'] = x_rfa[bins]
            print(f"    {tf}: done")

    np.savez_compressed(RFA_CACHE, **store)
    print(f"\nCached to {RFA_CACHE}")
    return store


def load_rfa_cache():
    if not os.path.exists(RFA_CACHE):
        return None
    return dict(np.load(RFA_CACHE, allow_pickle=True))


# ==================== DOSE-RESPONSE ====================

def auc_against_neg(y_pos_mask, y_neg_mask, scores):
    """AUC of scores separating positives (y_pos_mask) from negatives (y_neg_mask)."""
    m = y_pos_mask | y_neg_mask
    y = y_pos_mask[m].astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, 0, 0
    s = scores[m]
    try:
        return float(roc_auc_score(y, s)), int(y_pos_mask.sum()), int(y_neg_mask.sum())
    except ValueError:
        return np.nan, int(y_pos_mask.sum()), int(y_neg_mask.sum())


def perm_test_auc(y_bin, scores, n_perm=N_PERM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    auc_obs = float(roc_auc_score(y_bin, scores))
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = roc_auc_score(rng.permutation(y_bin), scores)
    p = float((null >= auc_obs).mean())
    return auc_obs, p, null


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = load_rfa_cache()
    if cache is None:
        cache = build_rfa_cache()
    else:
        print(f"Loaded RFA cache: {RFA_CACHE}")

    # aggregate across chromosomes
    rows = []
    for chrom in CHROMS:
        gids = cache[f'chr{chrom}_gene_id']
        for tf in TFS_TO_TEST:
            key = f'chr{chrom}_{tf}_rfa'
            if key not in cache:
                continue
            rfa = cache[key]
            for gid, eqtl_bin, min_p, n_sig, r in zip(
                gids,
                cache[f'chr{chrom}_eqtl_bin'],
                cache[f'chr{chrom}_min_p'],
                cache[f'chr{chrom}_n_sig'],
                rfa,
            ):
                rows.append({
                    'gene_id': gid, 'chrom': chrom, 'tf': tf,
                    'rfa': float(r),
                    'eqtl_bin': int(eqtl_bin),
                    'min_p': float(min_p),
                    'n_sig': int(n_sig),
                })

    df = pd.DataFrame(rows)
    print(f"\nAggregated: {len(df)} gene-TF-chrom rows, "
          f"{df['gene_id'].nunique()} unique genes, {df['tf'].nunique()} TFs")

    out_rows = []
    lines = ["=" * 120,
             "eQTL dose-response: does AUC(RFA) grow with eQTL strength?",
             f"genes: {df['gene_id'].nunique()}, TFs: {df['tf'].nunique()}",
             f"N_perm: {N_PERM}",
             "=" * 120, ""]

    for tf in TFS_TO_TEST:
        sub = df[df['tf'] == tf]
        if len(sub) < 50:
            continue
        neg_mask = (sub['eqtl_bin'] == 0).values
        pos_all = (sub['eqtl_bin'] == 1).values
        rfa = sub['rfa'].values

        # main AUC: all eQTL vs none
        auc_main, n_pos_all, n_neg = auc_against_neg(pos_all, neg_mask, rfa)
        # quantiles of -log10(min_p) among eQTL genes
        pos_df = sub[pos_all].copy()
        pos_df['strength'] = -np.log10(np.clip(pos_df['min_p'], 1e-300, 1.0))
        try:
            pos_df['q'] = pd.qcut(pos_df['strength'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        except ValueError:
            continue

        pos_positions = np.where(pos_all)[0]  # позиции eQTL-генов в sub
        for qlabel in ['Q1', 'Q2', 'Q3', 'Q4']:
            pos_mask = np.zeros(len(sub), dtype=bool)
            q_mask = (pos_df['q'] == qlabel).values
            pos_mask[pos_positions[q_mask]] = True
            auc_q, n_q, n_neg_q = auc_against_neg(pos_mask, neg_mask, rfa)
            out_rows.append({
                'tf': tf, 'analysis': 'strength_quantile', 'group': qlabel,
                'n_pos': n_q, 'n_neg': n_neg_q, 'auc': auc_q,
            })

        # bins of n_sig
        pos_df['nsig_bin'] = pd.cut(pos_df['n_sig'],
                                     bins=[0, 1, 2, 3, 5, np.inf],
                                     labels=['1', '2', '3', '4-5', '>=6'])
        for blabel in ['1', '2', '3', '4-5', '>=6']:
            pos_mask = np.zeros(len(sub), dtype=bool)
            b_mask = (pos_df['nsig_bin'] == blabel).values
            pos_mask[pos_positions[b_mask]] = True
            if pos_mask.sum() < 5:
                continue
            auc_b, n_b, n_neg_b = auc_against_neg(pos_mask, neg_mask, rfa)
            out_rows.append({
                'tf': tf, 'analysis': 'nsig_bin', 'group': blabel,
                'n_pos': n_b, 'n_neg': n_neg_b, 'auc': auc_b,
            })

        # permutation on main
        auc_obs, p_perm, _ = perm_test_auc(pos_all.astype(int), rfa)
        out_rows.append({
            'tf': tf, 'analysis': 'perm_main', 'group': 'all',
            'n_pos': int(pos_all.sum()), 'n_neg': int(neg_mask.sum()),
            'auc': auc_obs, 'p_perm': p_perm,
        })

    res = pd.DataFrame(out_rows)
    res.to_csv(os.path.join(OUT_DIR, "eqtl_dose_response.csv"), index=False)

    # --- summary table: strength quantiles ---
    lines.append("--- AUC(RFA) vs 'no eQTL' across strength quantiles (-log10 min_p) ---")
    lines.append(f"{'TF':<10} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'trend':>8} {'p_perm':>8}")
    lines.append("-" * 70)
    for tf in TFS_TO_TEST:
        sub = res[(res['tf'] == tf) & (res['analysis'] == 'strength_quantile')]
        if len(sub) < 4:
            continue
        aucs = sub.set_index('group')['auc'].reindex(['Q1', 'Q2', 'Q3', 'Q4']).values
        trend = aucs[-1] - aucs[0] if not np.isnan(aucs).any() else np.nan
        perm_row = res[(res['tf'] == tf) & (res['analysis'] == 'perm_main')]
        p_perm = perm_row['p_perm'].values[0] if len(perm_row) else np.nan
        lines.append(f"{tf:<10} " + " ".join(f"{a:>8.3f}" if not np.isnan(a) else "     -  " for a in aucs)
                     + f" {trend:>+8.3f}" if not np.isnan(trend) else "     -  "
                     + f" {p_perm:>8.3f}")

    lines.append("")
    lines.append("--- AUC(RFA) vs 'no eQTL' across n_sig bins ---")
    lines.append(f"{'TF':<10} {'1':>8} {'2':>8} {'3':>8} {'4-5':>8} {'>=6':>8}")
    lines.append("-" * 60)
    for tf in TFS_TO_TEST:
        sub = res[(res['tf'] == tf) & (res['analysis'] == 'nsig_bin')]
        if len(sub) < 2:
            continue
        aucs = sub.set_index('group')['auc'].reindex(['1', '2', '3', '4-5', '>=6']).values
        lines.append(f"{tf:<10} " + " ".join(
            f"{a:>8.3f}" if not np.isnan(a) else "     -  " for a in aucs))

    # summary verdict: mean trend across TFs
    lines.append("")
    lines.append("=" * 120)
    lines.append("VERDICT")
    lines.append("=" * 120)

    trend_vals = []
    for tf in TFS_TO_TEST:
        sub = res[(res['tf'] == tf) & (res['analysis'] == 'strength_quantile')]
        if len(sub) < 4:
            continue
        aucs = sub.set_index('group')['auc'].reindex(['Q1', 'Q2', 'Q3', 'Q4']).values
        if not np.isnan(aucs).any():
            trend_vals.append(aucs[-1] - aucs[0])
    if trend_vals:
        mean_trend = float(np.mean(trend_vals))
        lines.append(f"Mean AUC(Q4) - AUC(Q1) across TFs = {mean_trend:+.4f}")
        if mean_trend > 0.05:
            lines.append(">>> RFA AUC grows with eQTL strength: RFA captures eQTL effect size.")
        elif mean_trend < -0.05:
            lines.append(">>> RFA AUC drops with eQTL strength: unexpected; check sign.")
        else:
            lines.append(">>> RFA AUC is flat across eQTL strength: RFA captures 'has eQTL', "
                         "not 'strength of eQTL'.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "eqtl_dose_response.txt"), 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved:\n  {os.path.join(OUT_DIR, 'eqtl_dose_response.txt')}")
    print(f"  {os.path.join(OUT_DIR, 'eqtl_dose_response.csv')}")


if __name__ == '__main__':
    main()