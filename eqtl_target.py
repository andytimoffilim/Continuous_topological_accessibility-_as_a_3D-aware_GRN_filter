#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build binary eQTL target for genes from QTD000221 (GTEx v10 LCL, ge).
Streams the 2.4G file in chunks; per gene keeps min p-value and count of
variants with p < PVAL_THR. Saves eqtl_target.csv with columns:
  gene_id, min_p, n_sig, eqtl_bin
"""

import os
import pandas as pd
import numpy as np

EQTL_PATH   = r"E:\HIC\eqtl\QTD000221.all.tsv.gz"
OUT_CSV     = r"E:\HIC\eqtl\eqtl_target.csv"
PVAL_THR    = 5e-8          # genome-wide significance
CHUNK_ROWS  = 2_000_000
USECOLS     = ['molecular_trait_id', 'pvalue']

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    print(f"Streaming {EQTL_PATH} ...")

    min_p = {}   # gene_id -> min p-value
    n_sig = {}   # gene_id -> count of variants with p < PVAL_THR
    n_total = 0

    reader = pd.read_csv(
        EQTL_PATH, sep='\t', usecols=USECOLS,
        dtype={'molecular_trait_id': 'str', 'pvalue': 'float64'},
        chunksize=CHUNK_ROWS, compression='gzip',
        on_bad_lines='skip',
    )
    for i, chunk in enumerate(reader, 1):
        chunk = chunk.dropna(subset=['molecular_trait_id', 'pvalue'])
        g = chunk.groupby('molecular_trait_id')['pvalue'].agg(['min', 'size'])
        sig = chunk[chunk['pvalue'] < PVAL_THR] \
                .groupby('molecular_trait_id').size()
        for gid, row in g.iterrows():
            prev = min_p.get(gid, np.inf)
            if row['min'] < prev:
                min_p[gid] = row['min']
            n_sig[gid] = n_sig.get(gid, 0) + int(sig.get(gid, 0))
        n_total += len(chunk)
        print(f"  chunk {i}: rows so far = {n_total:,}, genes so far = {len(min_p):,}")

    df = pd.DataFrame({
        'gene_id': list(min_p.keys()),
        'min_p':   [min_p[g] for g in min_p],
    })
    df['n_sig'] = df['gene_id'].map(n_sig).fillna(0).astype(int)
    df['eqtl_bin'] = (df['n_sig'] > 0).astype(int)

    df = df.sort_values('min_p')
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"  genes total: {len(df):,}")
    print(f"  genes with >=1 sig variant (p<{PVAL_THR}): {int(df['eqtl_bin'].sum()):,}")
    print(f"  genes with 0 sig variants: {int((df['eqtl_bin']==0).sum()):,}")
    print(f"\nTop 5 genes by min_p:")
    print(df.head(5).to_string(index=False))


if __name__ == '__main__':
    main()