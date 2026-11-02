#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChIPSP-baseline: binary classification (peak -> loop -> gene) vs RFA.
Compares ROC-AUC of:
  - direct_contact (ChIPSP proxy, continuous score)
  - 1D distance (nearest peak, negative-signed)
  - RFA (gamma=0.85, with post-hoc PC1 residualization)
  - Combined (z-score RFA + z-score direct_contact)

Also reports:
  - Spearman(RFA, direct_contact) — is RFA just smoothed ChIPSP?
  - partial RFA | PC1 (binary)   — did residualization remove compartment signal?
  - weak-DC subgroup (Q10, Q25)  — does RFA win where direct contact is weak?
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, spilu
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
P_CACHE       = r"E:\HIC\results\pilot_multitf\P_matrix.npz"
A_CACHE       = r"E:\HIC\results\pilot_multitf\A_matrix.npz"
PC1_CACHE     = r"E:\HIC\results\pilot_multitf\pc1.npy"
MATRIX_CSV    = r"E:\genomes\atac_bigwig_results\hic_enhancer_analysis\matrix_with_hic_enhancers.csv"
GENE_STRUCT   = r"E:\genomes\gene_structures\gene_structure_hg38.csv"
BED_PATH      = r"E:\genomes\tfbs_GM12878.bed"
OUT_DIR       = r"E:\HIC\results\chips_baseline"

RESOLUTION    = 10_000
CHROM         = "chr1"
CHROM_PLAIN   = "1"
CELL_LINE     = "GM12878"

GAMMA_MAIN    = 0.85
N_BOOT        = 1000
RNG_SEED      = 42

TFS_TO_TEST = [
    'IKZF1', 'CTCF', 'SMC3', 'RAD21', 'PAX5', 'EBF1',
    'EP300', 'POLR2A', 'YY1', 'GABPA', 'MAX', 'E2F4',
    'ELF1', 'TCF12', 'BATF', 'BCL11A', 'IRF4', 'MEF2A',
    'NRF1', 'ZNF143',
]

DC_PERCENTILES = [50, 75, 90, 95, 99]
WEAK_QUANTILES = [10, 25]


# ==================== LOADERS ====================

def load_markov_cache():
    P = sp.load_npz(P_CACHE)
    A = sp.load_npz(A_CACHE)
    pc1 = np.load(PC1_CACHE)
    return P, A, pc1


def load_gene_promoters(path, chrom_plain):
    df = pd.read_csv(path, sep=',', low_memory=False)
    df = df[(df['element_type'] == 'promoter') &
            (df['chrom'].astype(str) == chrom_plain)]
    df = df.sort_values(['gene_id', 'start']).drop_duplicates('gene_id', keep='first')
    df['gene_id'] = df['gene_id'].astype(str).str.split('.').str[0]
    return df[['gene_id', 'start', 'end', 'gene_name']]


def peaks_to_vector(bed_path, chrom, tf, cell, resolution, n_bins):
    vec = np.zeros(n_bins)
    if not os.path.exists(bed_path):
        return vec
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6: continue
            if p[0] != chrom or p[3] != tf: continue
            if cell not in p[5]: continue
            b = int(p[1]) // resolution
            if 0 <= b < n_bins:
                vec[b] = 1.0
    return vec


def load_peaks_bp(bed_path, chrom, tf, cell):
    out = []
    if not os.path.exists(bed_path):
        return out
    with open(bed_path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 6: continue
            if p[0] != chrom or p[3] != tf: continue
            if cell not in p[5]: continue
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


def partial_spearman(x, y, z_list):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in z_list])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    return float(spearmanr(ex, ey)[0])


def roc_curve_data(y_true, y_score, n_points=200):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if len(fpr) > n_points:
        idx = np.linspace(0, len(fpr) - 1, n_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


# ==================== MAIN ====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print("ChIPSP-baseline vs RFA vs 1D distance (binary classification)")
    print("=" * 80)

    print("\nLoading P, A, pc1 from cache ...")
    P, A, pc1 = load_markov_cache()
    n = P.shape[0]
    print(f"  P: {P.shape}, nnz/n² = {P.nnz / (n * n):.4f}")

    print("\nLoading genes and features ...")
    genes = load_gene_promoters(GENE_STRUCT, CHROM_PLAIN)
    genes['bin'] = (genes['start'] // RESOLUTION).astype(int)
    genes = genes[(genes['bin'] >= 0) & (genes['bin'] < n)]

    feat = pd.read_csv(MATRIX_CSV, low_memory=False)
    feat['gene_id'] = feat['gene_id'].astype(str).str.split('.').str[0]
    need = ['gene_id'] + [tf for tf in TFS_TO_TEST if tf in feat.columns]
    merged_base = genes.merge(feat[need], on='gene_id', how='inner')
    bins = merged_base['bin'].astype(int).values
    print(f"  genes after merge: {len(merged_base)}")

    A_csr = A.tocsr()

    all_results = []
    all_curves = []

    for i, tf in enumerate(TFS_TO_TEST, 1):
        print(f"\n{'=' * 80}\n[{i}/{len(TFS_TO_TEST)}] {tf}\n{'=' * 80}")

        if tf not in merged_base.columns:
            print("  SKIP: no column")
            continue

        # --- 1. v0 из BED ---
        v0 = peaks_to_vector(BED_PATH, CHROM, tf, CELL_LINE, RESOLUTION, n)
        if v0.sum() == 0:
            print("  SKIP: no peaks")
            continue
        print(f"  v0 bins: {int(v0.sum())}")

        # --- 2. RFA ---
        x_rfa_raw = rfa_profile(P, v0, GAMMA_MAIN)
        print(f"  RFA(gamma={GAMMA_MAIN}) computed")

        # === STRONGER PC1 RESIDUALIZATION (post-hoc on x_rfa) ===
        X_pc1 = np.column_stack([np.ones(n), pc1])
        beta_pc1, *_ = np.linalg.lstsq(X_pc1, x_rfa_raw, rcond=None)
        x_rfa_resid = x_rfa_raw - X_pc1 @ beta_pc1
        x_rfa_resid = np.maximum(x_rfa_resid, 0)
        pc1_ratio = float(np.linalg.norm(x_rfa_resid) /
                          (np.linalg.norm(x_rfa_raw) + 1e-12))
        x_rfa = x_rfa_resid
        print(f"  PC1 post-hoc resid: "
              f"||raw||={np.linalg.norm(x_rfa_raw):.2e}, "
              f"||resid||={np.linalg.norm(x_rfa):.2e}, "
              f"ratio={pc1_ratio:.3f}")

        # --- 3. direct_contact ---
        peak_bins = np.where(v0 > 0)[0]
        A_sub = A_csr[:, peak_bins].tocsr()
        dc_full = np.asarray(A_sub.max(axis=1).todense()).flatten()
        merged = merged_base.copy()
        merged['direct_contact'] = dc_full[bins]
        merged['rfa'] = x_rfa[bins]

        # --- 4. target binary ---
        target_raw = merged[tf].values
        valid = ~(np.isnan(target_raw)
                  | np.isnan(merged['direct_contact'].values)
                  | np.isnan(merged['rfa'].values))
        y_raw = target_raw[valid]
        dc = merged['direct_contact'].values[valid]
        rfa = merged['rfa'].values[valid]
        n_valid = len(y_raw)

        y_bin = (y_raw > 0).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(n_valid - n_pos)
        print(f"  valid genes: {n_valid} (pos={n_pos}, neg={n_neg})")

        if n_pos < 10 or n_neg < 10:
            print("  SKIP: too few positives or negatives")
            continue

        # ==================== BLOCK 1: 1D distance baseline ====================
        peaks_bp = load_peaks_bp(BED_PATH, CHROM, tf, CELL_LINE)
        one_d_full = merged['start'].apply(
            lambda s: nearest_peak_distance(s, peaks_bp)).values
        x_1d = one_d_full[valid]
        auc_1d = float(roc_auc_score(y_bin, -x_1d))
        ci_1d = bootstrap_auc_ci(y_bin, -x_1d, N_BOOT)
        print(f"  AUC(1D dist) = {auc_1d:.3f}  CI [{ci_1d[0]:.3f}, {ci_1d[1]:.3f}]")

        # ==================== BLOCK 2: Spearman(RFA, dc) ====================
        rho_rfa_dc = float(spearmanr(rfa, dc)[0])
        print(f"  Spearman(RFA, direct_contact) = {rho_rfa_dc:+.3f}")
        if abs(rho_rfa_dc) > 0.8:
            print("  WARNING: RFA and direct_contact highly correlated — "
                  "RFA may be smoothed ChIPSP")

        # ==================== BLOCK 3: partial RFA | PC1 (binary) ====================
        pc1_bins = pc1[bins[valid]]
        part_pc1_bin = partial_spearman(rfa, y_bin, [pc1_bins])
        print(f"  partial RFA | PC1 (binary) = {part_pc1_bin:+.4f}")
        if part_pc1_bin > 0.2:
            print("  WARNING: RFA still correlates with PC1 — "
                  "residualization did not fully remove compartment signal")

        # ==================== AUCs ====================
        auc_chipsp = float(roc_auc_score(y_bin, dc))
        auc_rfa = float(roc_auc_score(y_bin, rfa))
        auc_rfa_neg = float(roc_auc_score(y_bin, -rfa))
        if auc_rfa_neg > auc_rfa:
            rfa_for_auc = -rfa
            auc_rfa = auc_rfa_neg
            rfa_sign_flipped = True
        else:
            rfa_for_auc = rfa
            rfa_sign_flipped = False

        z_rfa = (rfa_for_auc - rfa_for_auc.mean()) / (rfa_for_auc.std() + 1e-10)
        z_dc = (dc - dc.mean()) / (dc.std() + 1e-10)
        combined = z_rfa + z_dc
        auc_comb = float(roc_auc_score(y_bin, combined))

        ci_rfa = bootstrap_auc_ci(y_bin, rfa_for_auc, N_BOOT)
        ci_chip = bootstrap_auc_ci(y_bin, dc, N_BOOT)
        ci_comb = bootstrap_auc_ci(y_bin, combined, N_BOOT)

        delta_rfa_chip, d_lo, d_hi, p_delta = bootstrap_auc_diff(
            y_bin, rfa_for_auc, dc, N_BOOT)
        delta_rfa_1d = auc_rfa - auc_1d

        print(f"  AUC(ChIPSP) = {auc_chipsp:.3f}  CI [{ci_chip[0]:.3f}, {ci_chip[1]:.3f}]")
        print(f"  AUC(RFA)    = {auc_rfa:.3f}  CI [{ci_rfa[0]:.3f}, {ci_rfa[1]:.3f}]"
              f"{'  (sign flipped)' if rfa_sign_flipped else ''}")
        print(f"  AUC(Combined) = {auc_comb:.3f}  CI [{ci_comb[0]:.3f}, {ci_comb[1]:.3f}]")
        print(f"  ΔAUC (RFA - ChIPSP) = {delta_rfa_chip:+.4f}  "
              f"CI [{d_lo:+.4f}, {d_hi:+.4f}], p_boot={p_delta:.4f}")
        print(f"  ΔAUC (RFA - 1D)     = {delta_rfa_1d:+.4f}")

        # ==================== BLOCK 4: weak-DC subgroup ====================
        weak_results = {}
        for q in WEAK_QUANTILES:
            key = f'weak_q{q}'
            thr_dc_q = np.percentile(dc, q)
            weak = dc < thr_dc_q
            n_weak = int(weak.sum())
            weak_results[f'{key}_n'] = n_weak
            weak_results[f'{key}_auc_rfa'] = np.nan
            weak_results[f'{key}_auc_1d'] = np.nan
            weak_results[f'{key}_auc_chipsp'] = np.nan
            if n_weak < 30:
                print(f"  [weak DC Q{q}, n={n_weak}] too few, skipping")
                continue
            y_weak = y_bin[weak]
            if len(np.unique(y_weak)) < 2:
                print(f"  [weak DC Q{q}, n={n_weak}] single class, skipping")
                continue
            try:
                auc_rfa_w = float(roc_auc_score(y_weak, rfa_for_auc[weak]))
                auc_1d_w = float(roc_auc_score(y_weak, -x_1d[weak]))
                auc_chip_w = float(roc_auc_score(y_weak, dc[weak]))
            except ValueError:
                print(f"  [weak DC Q{q}, n={n_weak}] AUC failed, skipping")
                continue
            weak_results[f'{key}_auc_rfa'] = auc_rfa_w
            weak_results[f'{key}_auc_1d'] = auc_1d_w
            weak_results[f'{key}_auc_chipsp'] = auc_chip_w
            print(f"  [weak DC Q{q}, n={n_weak}, "
                  f"pos={int(y_weak.sum())}, neg={int((1 - y_weak).sum())}]")
            print(f"    AUC: RFA={auc_rfa_w:.3f}, 1D={auc_1d_w:.3f}, "
                  f"ChIPSP={auc_chip_w:.3f}")
            if auc_rfa_w > auc_1d_w:
                print(f"    >>> RFA beats 1D (+{auc_rfa_w - auc_1d_w:.3f})")
            else:
                print(f"    >>> 1D wins ({auc_rfa_w - auc_1d_w:+.3f})")

        # ==================== ChIPSP threshold sweep ====================
        chipsp_rows = []
        for pct in DC_PERCENTILES:
            thr = np.percentile(dc, pct)
            y_pred_chip = (dc > thr).astype(int)
            tp = int(((y_pred_chip == 1) & (y_bin == 1)).sum())
            fp = int(((y_pred_chip == 1) & (y_bin == 0)).sum())
            fn = int(((y_pred_chip == 0) & (y_bin == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            chipsp_rows.append({
                'pct': pct, 'threshold': float(thr),
                'n_pred_pos': int(y_pred_chip.sum()),
                'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
            })

        # ==================== ROC curves ====================
        fpr_rfa, tpr_rfa = roc_curve_data(y_bin, rfa_for_auc)
        fpr_chip, tpr_chip = roc_curve_data(y_bin, dc)
        fpr_comb, tpr_comb = roc_curve_data(y_bin, combined)
        fpr_1d, tpr_1d = roc_curve_data(y_bin, -x_1d)

        all_curves.append({
            'tf': tf,
            'fpr_rfa': fpr_rfa.tolist(), 'tpr_rfa': tpr_rfa.tolist(),
            'fpr_chip': fpr_chip.tolist(), 'tpr_chip': tpr_chip.tolist(),
            'fpr_comb': fpr_comb.tolist(), 'tpr_comb': tpr_comb.tolist(),
            'fpr_1d': fpr_1d.tolist(), 'tpr_1d': tpr_1d.tolist(),
        })

        # ==================== row ====================
        row = {
            'tf': tf,
            'n_genes': n_valid, 'n_pos': n_pos, 'n_neg': n_neg,
            'auc_1d': auc_1d,
            'auc_1d_lo': ci_1d[0], 'auc_1d_hi': ci_1d[1],
            'auc_chipsp': auc_chipsp,
            'auc_chipsp_lo': ci_chip[0], 'auc_chipsp_hi': ci_chip[1],
            'auc_rfa': auc_rfa,
            'auc_rfa_lo': ci_rfa[0], 'auc_rfa_hi': ci_rfa[1],
            'auc_combined': auc_comb,
            'auc_combined_lo': ci_comb[0], 'auc_combined_hi': ci_comb[1],
            'delta_rfa_minus_chipsp': delta_rfa_chip,
            'delta_rfa_chipsp_lo': d_lo, 'delta_rfa_chipsp_hi': d_hi,
            'p_boot_rfa_vs_chipsp': p_delta,
            'delta_rfa_minus_1d': delta_rfa_1d,
            'delta_1d_minus_chipsp': auc_1d - auc_chipsp,
            'rho_rfa_dc': rho_rfa_dc,
            'partial_rfa_pc1_binary': part_pc1_bin,
            'rfa_sign_flipped': rfa_sign_flipped,
            'rfa_pc1_resid_ratio': pc1_ratio,
        }
        for r in chipsp_rows:
            row[f'chipsp_pct{r["pct"]}_thr'] = r['threshold']
            row[f'chipsp_pct{r["pct"]}_f1'] = r['f1']
        row.update(weak_results)

        all_results.append(row)
        pd.DataFrame(all_results).to_csv(
            os.path.join(OUT_DIR, "chips_baseline_results.csv"), index=False)

    # ==================== SUMMARY ====================
    res = pd.DataFrame(all_results)
    res.to_csv(os.path.join(OUT_DIR, "chips_baseline_results.csv"), index=False)

    def _fmt(v):
        return f"{v:.3f}" if pd.notna(v) else "  -  "

    lines = [
        "=" * 160,
        f"ChIPSP / 1D / RFA / Combined — {len(res)} TFs",
        f"GM12878 chr1 10kb | gamma={GAMMA_MAIN} | N_boot={N_BOOT}",
        "=" * 160, "",
        f"{'TF':<10} {'N':<5} {'AUC_1D':>8} {'AUC_Chip':>9} {'AUC_RFA':>8} "
        f"{'dAUC(RFA-Chip)':>15} {'dAUC(RFA-1D)':>13} "
        f"{'rho(RFA,dc)':>12} {'part|PC1':>9} "
        f"{'w10_RFA':>8} {'w10_1D':>7} {'w25_RFA':>8} {'w25_1D':>7}",
        "-" * 160,
    ]
    for _, r in res.sort_values('delta_rfa_minus_chipsp', ascending=False).iterrows():
        sig = ""
        if r['delta_rfa_chipsp_lo'] > 0:
            sig = " ***" if r['p_boot_rfa_vs_chipsp'] < 0.01 else \
                  " **" if r['p_boot_rfa_vs_chipsp'] < 0.05 else " *"
        lines.append(
            f"{r['tf']:<10} {int(r['n_genes']):<5} "
            f"{r['auc_1d']:>8.3f} {r['auc_chipsp']:>9.3f} {r['auc_rfa']:>8.3f} "
            f"{r['delta_rfa_minus_chipsp']:>+15.4f}{sig:<4} "
            f"{r['delta_rfa_minus_1d']:>+13.4f} "
            f"{r['rho_rfa_dc']:>+12.3f} {r['partial_rfa_pc1_binary']:>+9.4f} "
            f"{_fmt(r.get('weak_q10_auc_rfa')):>8} {_fmt(r.get('weak_q10_auc_1d')):>7} "
            f"{_fmt(r.get('weak_q25_auc_rfa')):>8} {_fmt(r.get('weak_q25_auc_1d')):>7}"
        )

    n_rfa_better_chip = int((res['delta_rfa_minus_chipsp'] > 0).sum())
    n_sig_chip = int((res['delta_rfa_chipsp_lo'] > 0).sum())
    n_rfa_better_1d = int((res['delta_rfa_minus_1d'] > 0).sum())
    n_high_corr = int((res['rho_rfa_dc'].abs() > 0.8).sum())
    n_high_pc1 = int((res['partial_rfa_pc1_binary'] > 0.2).sum())

    w10_rfa = res['weak_q10_auc_rfa'].dropna()
    w10_1d = res['weak_q10_auc_1d'].dropna()
    w25_rfa = res['weak_q25_auc_rfa'].dropna()
    w25_1d = res['weak_q25_auc_1d'].dropna()

    lines += [
        "", "=" * 160, "AGGREGATE", "=" * 160,
        f"Mean AUC 1D     : {res['auc_1d'].mean():.3f}",
        f"Mean AUC ChIPSP : {res['auc_chipsp'].mean():.3f}",
        f"Mean AUC RFA    : {res['auc_rfa'].mean():.3f}",
        f"Mean AUC Comb   : {res['auc_combined'].mean():.3f}",
        "",
        f"Mean dAUC(RFA - ChIPSP): {res['delta_rfa_minus_chipsp'].mean():+.4f}",
        f"Mean dAUC(RFA - 1D)    : {res['delta_rfa_minus_1d'].mean():+.4f}",
        "",
        f"RFA > ChIPSP (any)      : {n_rfa_better_chip}/{len(res)}",
        f"RFA > ChIPSP (CI > 0)   : {n_sig_chip}/{len(res)}  <- main claim",
        f"RFA > 1D                : {n_rfa_better_1d}/{len(res)}",
        "",
        f"Mean Spearman(RFA, dc)  : {res['rho_rfa_dc'].mean():+.3f}",
        f"TFs with |rho(RFA,dc)| > 0.8: {n_high_corr}/{len(res)}",
        f"Mean partial RFA | PC1 (binary): {res['partial_rfa_pc1_binary'].mean():+.4f}",
        f"TFs with partial RFA | PC1 > 0.2: {n_high_pc1}/{len(res)}",
        f"Mean PC1 resid ratio ||x_resid||/||x_raw||: {res['rfa_pc1_resid_ratio'].mean():.3f}",
        "",
    ]

    if len(w10_rfa) > 0:
        n_w10 = int((w10_rfa.values > w10_1d.values[:len(w10_rfa)]).sum())
        lines += [
            "--- Weak-DC subgroups ---",
            f"Q10: n_TFs={len(w10_rfa)}, "
            f"mean AUC RFA={w10_rfa.mean():.3f}, 1D={w10_1d.mean():.3f}",
            f"Q10: TFs where RFA > 1D: {n_w10}/{len(w10_rfa)}",
        ]
    if len(w25_rfa) > 0:
        n_w25 = int((w25_rfa.values > w25_1d.values[:len(w25_rfa)]).sum())
        lines += [
            f"Q25: n_TFs={len(w25_rfa)}, "
            f"mean AUC RFA={w25_rfa.mean():.3f}, 1D={w25_1d.mean():.3f}",
            f"Q25: TFs where RFA > 1D: {n_w25}/{len(w25_rfa)}",
        ]

    lines.append("")

    if n_sig_chip >= 0.6 * len(res):
        lines.append(">>> RFA SIGNIFICANTLY BEATS ChIPSP for majority of TFs.")
    elif n_sig_chip >= 0.3 * len(res):
        lines.append(">>> MIXED: RFA beats ChIPSP for some TFs.")
    else:
        lines.append(">>> RFA does NOT beat ChIPSP for majority.")

    if n_rfa_better_1d >= 0.6 * len(res):
        lines.append(">>> RFA also beats 1D distance for majority of TFs.")
    else:
        lines.append(">>> NOTE: 1D distance is competitive — position RFA advantage carefully.")

    if n_high_corr > 0.3 * len(res):
        lines.append(">>> WARNING: RFA highly correlated with direct_contact for many TFs.")

    if len(w10_rfa) > 0 and int((w10_rfa.values > w10_1d.values[:len(w10_rfa)]).sum()) >= 0.3 * len(w10_rfa):
        lines.append(">>> NICHE FOUND: RFA beats 1D in weak-contact subgroup for >=30% of TFs.")
    else:
        lines.append(">>> NO NICHE: 1D still wins even in weak-contact subgroup.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "chips_baseline_summary.txt"), 'w') as f:
        f.write(text)

    with open(os.path.join(OUT_DIR, "roc_curves.json"), 'w') as f:
        json.dump(all_curves, f)

    print(f"\nSaved:\n  {os.path.join(OUT_DIR, 'chips_baseline_results.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'chips_baseline_summary.txt')}")
    print(f"  {os.path.join(OUT_DIR, 'roc_curves.json')}")


if __name__ == '__main__':
    main()