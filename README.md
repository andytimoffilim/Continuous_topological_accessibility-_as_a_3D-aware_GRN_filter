# Continuous topological accessibility (RFA) as a 3D-aware GRN filter

Code and data for the manuscript:

> **Continuous topological accessibility (RFA) as a 3D-aware GRN filter:**
> **anticorrelation with eQTL and no prediction of ΔRNA**
> Timofeev Andrey, AI Center for SCO+ Countries, Saint Petersburg, Russia (2026)

## Overview

RFA (Relative Forest Accessibility) is a continuous topological accessibility
index computed as the solution of `(I - γP) x = v₀`, where `P` is a Markov
transition matrix built from Hi-C contacts and `v₀` is a vector of TF peaks
(or ATAC peaks). RFA yields a continuous accessibility profile across all
genomic bins.

Main results:
- **Promoter count** (GM12878 chr1, 10 kb, 811 genes): RFA loses to 1D
  (AUC 0.836 vs 0.976). Methodological artifact: target and 1D distance
  look at the same window.
- **eQTL** (GTEx v10 LCL, chr1+17+19, 10 kb, 1625 genes): RFA
  **anticorrelates** with eQTL. AUC(RFA) = 0.386 (below chance),
  AUC(−RFA) = 0.614. Dose-response trend = −0.056 (19/20 TFs).
- **Topology controls** (chr1, 10 kb, 5 TFs): a distance-preserving shuffle
  of the Hi-C matrix reproduces the anticorrelation (ΔAUC vs real:
  [−0.074, +0.076], all CIs include 0), whereas a degree-preserving
  rewiring destroys it (ΔAUC ∈ [−0.261, −0.003], CIs exclude 0 for 5/5 TFs).
  **The anticorrelation is carried by distance-decay (local 3D accessibility),
  not by higher-order topology.**
- **Topology controls chr17/19** (10 kb, 5 TFs, ρ-primary): chr17
  replicates the chr1 mechanism (rewired Δρ = −0.13, 3/5 TFs; shuffled
  0/5); chr19 is inverted (shuffled 4/5, rewired 0/5), consistent with
  local within-diagonal heterogeneity on a short gene-dense chromosome.
- **ΔRNA** (MCF10A vs MCF7, 40 kb, ATAC peaks, 1730 genes): RFA does **not**
  predict ΔRNA (ρ = +0.0047, perm p = 0.866, bootstrap 95% CI [−0.040, +0.053],
  MDR = 0.067). CI includes zero and does not exclude the practical boundary
  |ρ| = 0.05.

## Repository structure
.
├── README.md
├── CHANGELOG.md # baseline → after-audit
├── LICENSE
├── full_version.tex # manuscript (single-file LaTeX)
├── converter.py # Hi-C .txt → .npy
├── MULTI-TF PILOT.py # builds P, RFA/1D/dc, chr1
├── chips_baseline_full.py # binary target, AUC, weak-DC, pc1_ratio
├── eqtl_target.py # GTEx v10 LCL target
├── eqtl_multi_chr.py # eQTL chr1+17+19 (WITH sign flip — see caveat)
├── eqtl_dose_response.py # dose-response, NO flip (authoritative)
├── delta_rfa_vs_delta_rna.py # ΔRNA main test
├── delta_rfa_negative_control.py # ΔRNA negative control + power
├── step1_bootstrap_delta_rna.py # bootstrap CI for ΔRNA (audit step 1)
├── step2_operator_audit.py # P/A/gamma-limit/variance audit (step 2)
├── step3_topology_controls.py # degree-rewire/shuffle/flatten, chr1 (step 3)
├── step3_5_topology_controls_chr17_19.py # chr17/19 replication, ρ-primary (step 3.5)
├── data/ # (not included; paths below)
└── results/
├── pilot_multitf/
├── chips_baseline/
├── eqtl_multi_chr/
├── eqtl_dose/
├── delta_rfa/
│ ├── delta_rfa_results.csv
│ ├── delta_rfa_summary.txt
│ ├── delta_rfa_negative_control.txt
│ └── delta_rna_ci.csv # step 1 output
├── operator_audit/ # step 2 outputs
│ ├── operator_audit.txt
│ ├── operator_audit_P.csv
│ ├── operator_audit_A.csv
│ ├── operator_audit_gamma_limit.csv
│ ├── operator_audit_var.csv
│ └── operator_audit_pc1.csv
└── topology_controls/ # step 3 / 3.5 outputs
├── topology_controls_chr1_5tf.csv
├── topology_controls_chr1_5tf.txt
├── topology_controls_chr17_5tf.csv
├── topology_controls_chr17_5tf.txt
├── topology_controls_chr19_5tf.csv
├── topology_controls_chr19_5tf.txt
├── topology_controls_chr17_19_5tf.csv # combined
└── topology_controls_chr17_19_5tf.txt

text

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `matplotlib` (for figures)
- `juicer_tools.2.20.00.ac.jar` (for Hi-C extraction)

No `networkx`/`igraph` required: the topology controls use a pure-numpy
`double_edge_swap`, distance-preserving shuffle, and configuration-model
flattening.

## Data

All data paths below are absolute; adjust to your environment.

### Hi-C

| File | Resolution | Chromosomes | Source |
|---|---|---|---|
| `GM12878_chr{1,17,19}_10kb.npy` | 10 kb | 1, 17, 19 | ENCODE/4DN `4DNFIZGCA8AH.hic` |
| `MCF{10A,7}_chr{1,17,19}_40kb_fixed.npy` | 40 kb | 1, 17, 19 | Provided (suffix `_fixed` = symmetrization / diagonal fix) |

Extraction for GM12878:

```bash
juicer_tools dump observed NONE 4DNFIZGCA8AH.hic 1 1 BP 10000 GM12878_chr1_10kb.txt
juicer_tools dump observed NONE 4DNFIZGCA8AH.hic 17 17 BP 10000 GM12878_chr17_10kb.txt
juicer_tools dump observed NONE 4DNFIZGCA8AH.hic 19 19 BP 10000 GM12878_chr19_10kb.txt
Then converter.py for each .txt → .npy.

TFBS (GM12878)
E:\genomes\tfbs_GM12878.bed (BED6). 20 TFs: IKZF1, CTCF, SMC3, RAD21,
PAX5, EBF1, EP300, POLR2A, YY1, GABPA, MAX, E2F4, ELF1, TCF12, BATF,
BCL11A, IRF4, MEF2A, NRF1, ZNF143.

ATAC peaks (MCF10A / MCF7)
MCF7: GSE169929_ENCFF882OVP_IDR_thresholded_peaks_GRCh38.bed.gz
(GEO GSE169929)

MCF10A: GSM3735772_MCF10A_ATAC_peaks.broadPeak.gz (GEO GSM3735772)

Only the first three columns (chrom, start, end) are used; v₀ = bins
containing an ATAC peak.

eQTL (GTEx v10 LCL)
QTD000221.all.tsv.gz from the eQTL Catalogue (EBI), study QTS000015,
dataset QTD000221, GTEx v10 LCL, quant_method=ge, sample_size=326.

RNA (MCF10A / MCF7)
MCF10A_RNA.tsv (16 829 genes), MCF7_RNA.tsv (19 154 genes);
columns gene_id, gene_name, TPM.

Gene annotation
gene_structure_hg38.csv (element_type=='promoter', first promoter
per gene, chrom without chr).

Reproducing the results
Step 1: Promoter count (GM12878 chr1, 10 kb)
bash
python "MULTI-TF PILOT.py"           # builds P, computes RFA/1D/dc
python chips_baseline_full.py        # binary target, AUC, weak-DC subgroup
Outputs:

results/pilot_multitf/P_matrix.npz (cached P)

results/chips_baseline/chips_baseline_summary.txt

results/chips_baseline/chips_baseline_results.csv

Step 2: eQTL target
bash
python eqtl_target.py
Output: eqtl/eqtl_target.csv (18 988 genes, 1 567 with eQTL).

Step 3: eQTL multi-chromosome (chr1+17+19)
bash
python eqtl_multi_chr.py
Outputs:

results/eqtl_multi_chr/P_chr{17,19}.npz (cached P)

results/eqtl_multi_chr/eqtl_multi_chr_summary.txt

Warning: the AUC values in eqtl_multi_chr_summary.txt are sign-flipped
(AUC = 0.611 for RFA). The correct values (no flip) are
AUC(RFA) = 0.386, AUC(−RFA) = 0.614; see Step 4.

Step 4: eQTL dose-response (no flip)
bash
python eqtl_dose_response.py
Outputs:

results/eqtl_dose/eqtl_rfa_cache.npz (RFA per TF per gene)

results/eqtl_dose/eqtl_dose_response.csv

results/eqtl_dose/eqtl_dose_response.txt

This is the authoritative eQTL analysis (no sign flip).

Step 5: ΔRNA (MCF10A vs MCF7, 40 kb)
bash
python delta_rfa_vs_delta_rna.py
python delta_rfa_negative_control.py
Outputs:

results/delta_rfa/delta_rfa_results.csv

results/delta_rfa/delta_rfa_summary.txt

results/delta_rfa/delta_rfa_negative_control.txt

Step 6 (audit): bootstrap CI for ΔRNA
bash
python step1_bootstrap_delta_rna.py
Output: results/delta_rfa/delta_rna_ci.csv (bootstrap 95% CI for ρ,
per-chromosome and within-line; B = 1000, gene-level resampling, seed = 42).

Key result: ρ = +0.0047, perm p = 0.866, bootstrap CI [−0.040, +0.053].
The CI includes zero and does not exclude the practical boundary |ρ| = 0.05.

Step 7 (audit): operator audit
bash
python step2_operator_audit.py
Outputs (in results/operator_audit/):

operator_audit.txt (human-readable report)

operator_audit_P.csv (P stochasticity, non-negativity, zero rows)

operator_audit_A.csv (A symmetry, cached vs raw)

operator_audit_gamma_limit.csv (γ → 1 limit)

operator_audit_var.csv (Var[x_γ] / Var[v₀] on γ grid)

operator_audit_pc1.csv (PC1 residualization ratio)

Key findings:

P has zero rows (8.6% chr1, 3.7% chr17, 5.1% chr19) → self-loop
correction P[i,i]=1 for d_i=0 is required.

Var[x_γ]/Var[v₀] grows with γ (1.7 at 0.5, 8.4 at 0.85, 684 at 0.99).
The "contraction-like" claim is false; the correct interpretation is
amplification along the stationary mode.

A^res is not symmetric (max|A−Aᵀ| = 12.5 on chr1); P may have
complex eigenvalues.

The γ → 1 limit is a constant (π·v₀)/(π·1)·1, not π.

rfa_pc1_resid_ratio = 0.623 is ‖x_resid‖₂/‖x_raw‖₂ on all bins, from
chips_baseline_full.py (promoter benchmark only). It does not apply to
the eQTL pipeline.

Step 8 (audit): topology controls chr1
bash
python step3_topology_controls.py
Outputs (in results/topology_controls/):

topology_controls_chr1_5tf.csv

topology_controls_chr1_5tf.txt

Design: real Hi-C vs three controls on chr1, 10 kb, 5 TFs (IKZF1, CTCF,
EP300, POLR2A, YY1). All graphs are passed through the same pipeline
(PC1 residualization, non-negative clipping, self-loop on zero-degree bins,
P = D⁻¹A, γ = 0.85).

Controls:

Degree-preserving rewiring (double_edge_swap, 10×nnz swaps):
destroys the anticorrelation. ΔAUC ∈ [−0.261, −0.003], CIs exclude 0 for
5/5 TFs.

Distance-preserving shuffle (permute values within each diagonal
|i−j| = d): reproduces the anticorrelation. ΔAUC ∈ [−0.074, +0.076], all
CIs include 0 for 5/5 TFs.

Flattened graph (configuration model with the same degree sequence):
intermediate. ΔAUC ∈ [−0.202, +0.033], significant only for EP300.

Conclusion: the eQTL anticorrelation is carried by distance-decay
(local 3D accessibility), not by higher-order topology (loops, TADs,
compartments).

Step 9 (audit): topology controls chr17/19
bash
python step3_5_topology_controls_chr17_19.py
Outputs (in results/topology_controls/):

topology_controls_chr17_5tf.csv, topology_controls_chr19_5tf.csv

topology_controls_chr17_19_5tf.csv, topology_controls_chr17_19_5tf.txt

Design: same as step 8, extended to chr17 and chr19, 5 TFs,
N_SWAP_FACTOR=10. Because n_pos ≈ 30–50 per chromosome, AUC-based Δ
statistics are underpowered (MDR(ΔAUC) ≈ 0.10–0.15 vs observed
ΔAUC ≈ 0.05–0.07); ρ is the primary metric on these chromosomes.

Key results:

chr17 replicates the chr1 mechanism (by ρ):
rewired kills the anticorrelation (Δρ = −0.125/−0.125/−0.130 for
IKZF1/POLR2A/YY1, 3/5 TFs, CIs exclude 0; mean ρ shifts from −0.106
to −0.004, shift +0.101); shuffled preserves it (0/5 significant,
mean ρ = −0.061, shift +0.045). Rewired shift is 2.3× larger than
shuffled shift.

chr19 shows an inverted pattern: distance-preserving shuffle
removes the anticorrelation (4/5 TFs significant; mean ρ shifts from
−0.120 to −0.058, shift +0.063), while rewiring preserves it (0/5;
mean ρ = −0.099, shift +0.021). Consistent with local within-diagonal
heterogeneity as the carrier on short gene-dense chromosomes
(chr19: 5861 bins vs 24895 on chr1).

Key parameters
γ = 0.85 (fixed; not tuned on test)

top-k = 300 (row sparsification)

PC1 residualization on A (before P = D⁻¹A)

Self-loop P[i,i] = 1 for zero-degree bins (d_i = 0) — applied to
real P and to all controls in steps 3 / 3.5

Rewiring factor: 10×nnz swaps for chr1, chr17, chr19 (steps 3, 3.5)

Resolution: 10 kb (GM12878) / 40 kb (MCF)

Bootstrap: 1000 resamples (gene-level, seed = 42)

Permutation: 500 permutations

Minimum genes per analysis: 20–50

Important caveats
Sign flip. eqtl_multi_chr.py applies
rfa_for_auc = -rfa if auc_rfa_neg > auc_rfa else rfa, yielding
AUC = 0.611. Without the flip: AUC(RFA) = 0.386, AUC(−RFA) = 0.614.
The final dose-response analysis (eqtl_dose_response.py) removes the
flip and reports both directions.

Self-loop on zero-degree bins. After top-k sparsification, PC1
residualization, and non-negative clipping, 8.6% (chr1), 3.7% (chr17),
5.1% (chr19) of bins have no outgoing contacts. The pipeline adds
self-loops to make P row-stochastic. Results with and without
self-loops differ by < 0.03 in AUC on the tested targets; the manuscript
reports the no-self-loop values and states the correction in Methods.

eQTL target vs. genetic layer. eQTL is a genetic layer; RFA is
epigenetic. The anticorrelation may reflect layer mismatch, not biology.
Three hypotheses are discussed in the manuscript (§6.3); the shuffle
control in step 3 weakens the pure layer-mismatch interpretation.

Resolution dependence. RFA works at 10 kb with TF peaks; it does not
work at 40 kb with ATAC peaks (ρ ≈ 0, MDR = 0.067). The signal lives at
fine scales and is sensitive to the type of v₀.

AUC underpowered on chr17/19. With n_pos ≈ 30–50 per chromosome,
ΔAUC CIs are wide (±0.10) and cannot resolve the observed
ΔAUC ≈ 0.05–0.07. We report ρ as the primary metric on chr17/19;
AUC is supporting only. See step 9 and §5.2.2 of the manuscript.

PC1 residualization is applied at two levels. (a) Row-wise on the
Hi-C matrix A inside build_markov_matrix (common to all analyses);
(b) post-hoc on the RFA profile x_γ in chips_baseline_full.py, where
‖x_resid‖₂/‖x_raw‖₂ = 0.623 on average on the promoter-count benchmark.
The number 0.623 is specific to that benchmark; the eQTL pipeline does not
apply post-hoc residualization to x_γ. See §6.4 of the manuscript.

Audit trail
This repository includes the baseline pipeline (steps 1–5) and the
audit scripts (steps 6–9). The audit was performed in response to the
plan in Идея 1. Зафиксировать интерпретацию оператора до.txt (not
included). The manuscript full_version.tex reflects the post-audit
state: bootstrap CI for ΔRNA, corrected operator properties (self-loop,
non-contraction, γ → 1 limit), topology controls chr1 (distance-decay,
not loops), and chr17/19 replication (chr17 replicates, chr19 inverted).

See CHANGELOG.md for a version-by-version summary.

Planned / future work
20-TF replication of the topology controls on chr17/19 (currently 5 TFs).

enhancer_gene_benchmark.py — CRISPRi benchmark (Gasperini 2019 / Fulco 2019). Separate project.

pc1_ratio_eqtl.py — PC1 residualization ratio on eQTL genes. Not
needed: 0.623 is tied to the promoter benchmark; the eQTL diagnostic
is partial RFA | PC1 = −0.03.

Citation
If you use this code, please cite:

text
Timofeev, A. (2026). Continuous topological accessibility (RFA) as a 3D-aware
GRN filter: anticorrelation with eQTL and no prediction of ΔRNA.
Manuscript in preparation.
License
MIT (see LICENSE).

Contact
Andrey Timofeev, AI Center for SCO+ Countries, Saint Petersburg, Russia.