# C:\Users\user\PycharmProjects\TAD\step1_bootstrap_delta_rna.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN_CSV  = r"E:\HIC\results\delta_rfa\delta_rfa_results.csv"
OUT_CSV = r"E:\HIC\results\delta_rfa\delta_rna_ci.csv"

df = pd.read_csv(IN_CSV)
print(f"loaded {len(df)} rows from {IN_CSV}")
print("columns:", list(df.columns))

n = len(df)
assert n == 1730, f"expected 1730 genes, got {n}"

rng = np.random.default_rng(42)
B = 1000

def boot_ci(x, y, B=1000, rng=None):
    n = len(x)
    rhos = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        rhos[b] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return lo, hi, rhos

rows = []

# --- all ---
x, y = df["dRFA"].values, df["dRNA"].values
r = spearmanr(x, y).statistic
lo, hi, _ = boot_ci(x, y, B, rng)
rows.append(dict(subset="all", n=n, rho=r, ci_lo=lo, ci_hi=hi))

# --- per chromosome ---
for chrom, sub in df.groupby("chrom"):
    xs, ys = sub["dRFA"].values, sub["dRNA"].values
    r = spearmanr(xs, ys).statistic
    lo, hi, _ = boot_ci(xs, ys, B, rng)
    rows.append(dict(subset=f"chr{chrom}", n=len(xs), rho=r, ci_lo=lo, ci_hi=hi))

# --- within-line sanity ---
for col_rfa, col_rna, name in [("RFA_10A", "log10A", "within_10A"),
                               ("RFA_7",  "log7",  "within_7")]:
    xs, ys = df[col_rfa].values, df[col_rna].values
    r = spearmanr(xs, ys).statistic
    lo, hi, _ = boot_ci(xs, ys, B, rng)
    rows.append(dict(subset=name, n=len(xs), rho=r, ci_lo=lo, ci_hi=hi))

res = pd.DataFrame(rows)
res["mdr"] = 0.0673
res["boundary"] = 0.05
res["ci_includes_boundary"] = (res.ci_lo <= res.boundary) & (res.boundary <= res.ci_hi)
res.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}")
print(res.to_string(index=False))