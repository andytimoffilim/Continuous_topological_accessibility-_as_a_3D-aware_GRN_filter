import numpy as np
import pandas as pd

print("Loading triplets ...")
df = pd.read_csv(r"E:\HIC\results\GM12878_chr1_10kb.txt",
                 sep='\t', header=None, names=['i', 'j', 'c'])
print(f"triplets: {len(df):,}")

is_float = (df['c'] % 1 != 0).any()
dtype = np.float32 if is_float else np.int32
print(f"fractional counts: {is_float} → dtype={dtype.__name__}")

n = max(df['i'].max(), df['j'].max()) // 10000 + 1
print(f"n bins: {n}")

M = np.zeros((n, n), dtype=dtype)
i_bin = (df['i'] // 10000).astype(int).values
j_bin = (df['j'] // 10000).astype(int).values
for a, b, c in zip(i_bin, j_bin, df['c'].values):
    M[a, b] = c
    M[b, a] = c

np.save(r"E:\HIC\results\GM12878_chr1_10kb.npy", M)
print(f"saved: {M.shape}, dtype={M.dtype}")
print(f"nnz/n² = {(M > 0).mean():.4f}")
print(f"max: {M.max():.2f}, mean: {M.mean():.4f}")