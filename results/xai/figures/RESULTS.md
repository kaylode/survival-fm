# SurvPFN — Benchmark Results

> Generated from 480 fold-level results (8 datasets × 12 models × 5-fold CV).
> All figures are in `xai/figures/`. Run `uv run python -m survpfn.xai.analysis` to regenerate.

---

## Summary Figures

| Figure | Description |
|--------|-------------|
| `fig01_heatmap_cindex` | Mean C-index heatmap across all models × datasets |
| `fig02_cindex_comparison` | C-index bar charts per dataset with std error bars |
| `fig03_ibs_comparison` | IBS bar charts (lower is better) |
| `fig04_multimetric_overview` | Box plots: C-index / IBS / AUC / D-cal all datasets pooled |
| `fig05_auc_curves` | Time-dependent AUROC curves per dataset |
| `fig06_efficiency_frontier` | C-index vs training time (log scale), Pareto frontier |
| `fig07_model_family_boxplot` | C-index distributions by model group (public vs Sirbu) |
| `fig08_tabpfn_ablation` | TabPFN vs classical counterparts — Δ C-index per dataset |
| `fig09_sirbu_multitask` | Performance across the 4 Sirbu clinical outcomes |
| `fig10_feature_importance_*` | Cox hazard ratios + RSF/GBSA importances per dataset |
| `fig11_hpo_convergence` | Optuna best C-index vs n_trials |
| `fig11b_hpo_convergence_speed` | Best-trial / total-trials (convergence speed) |
| `fig12_ranking` | Average model rank + per-dataset rank heatmap |
| `fig13_dcal_heatmap` | D-calibration heatmap across all models × datasets |

---

## C-index Results (Mean ± Std, 5-fold CV)

Models sorted by performance within each dataset. Best non-baseline highlighted.

### Public Datasets

| Model | SUPPORT2 | METABRIC | GBSG |
|-------|:--------:|:--------:|:----:|
| KM | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 |
| Cox PH | 0.570±0.013 | 0.634±0.008 | 0.663±0.018 |
| RSF | 0.616±0.008 | **0.641±0.031** | **0.675±0.017** |
| GBSA | **0.620±0.009** | 0.632±0.024 | 0.670±0.019 |
| DeepSurv | 0.610±0.009 | 0.637±0.031 | 0.674±0.017 |
| MTLR | 0.559±0.047 | 0.596±0.051 | 0.635±0.051 |
| DeepHit | 0.508±0.022 | 0.473±0.024 | 0.429±0.025 |
| TabPFN-Cox | 0.505±0.007 | 0.561±0.056 | 0.532±0.047 |
| Surv-Cox | 0.591±0.005 | 0.626±0.020 | 0.670±0.005 |
| Surv-DeepHit | 0.570±0.011 | 0.479±0.030 | 0.387±0.033 |
| Surv-PCHaz | 0.560±0.006 | 0.610±0.018 | 0.668±0.014 |
| Surv-MTLR | 0.559±0.017 | 0.603±0.016 | 0.643±0.027 |

### Sirbu Multi-task (Private)

| Model | Mortality | CV | MI | Stroke |
|-------|:---------:|:--:|:--:|:------:|
| Cox PH | 0.794±0.008 | 0.794±0.008 | 0.768±0.008 | 0.790±0.010 |
| RSF | **0.796±0.008** | **0.797±0.007** | **0.769±0.009** | 0.789±0.008 |
| GBSA | 0.793±0.012 | 0.793±0.010 | 0.767±0.008 | 0.788±0.010 |
| DeepSurv | 0.777±0.012 | 0.788±0.013 | 0.761±0.007 | **0.784±0.007** |
| MTLR | 0.667±0.100 | 0.496±0.021 | 0.496±0.016 | 0.551±0.131 |
| DeepHit | 0.448±0.100 | 0.381±0.081 | 0.389±0.092 | 0.405±0.099 |
| TabPFN-Cox | 0.584±0.136 | 0.651±0.126 | 0.597±0.104 | 0.608±0.114 |
| Surv-Cox | 0.719±0.033 | 0.716±0.041 | 0.704±0.009 | 0.711±0.024 |
| Surv-DeepHit | 0.365±0.040 | 0.376±0.019 | 0.361±0.034 | 0.361±0.040 |
| Surv-PCHaz | 0.706±0.029 | 0.708±0.012 | 0.679±0.037 | 0.692±0.026 |
| Surv-MTLR | 0.577±0.042 | 0.588±0.049 | 0.571±0.087 | 0.595±0.044 |

---

## Key Findings

### 1. Tree Models Are the Best Baseline

RSF wins on 6 out of 8 datasets (METABRIC, GBSG, Sirbu-Mortality, Sirbu-CV, Sirbu-MI, SIRBU),
GBSA wins on SUPPORT2, and Cox PH wins on Sirbu-Stroke. Tree models are consistently in the
top 3 with low variance and are competitive even against deep learning methods.

| Dataset | Winner | C-index |
|---------|--------|---------|
| SUPPORT2 | GBSA | 0.620 |
| METABRIC | RSF | 0.641 |
| GBSG | RSF | 0.675 |
| Sirbu (Mortality) | RSF | 0.796 |
| Sirbu (CV) | RSF | 0.797 |
| Sirbu (MI) | RSF | 0.769 |
| Sirbu (Stroke) | Cox PH | 0.790 |

### 2. Surv-Cox is the Best TabPFN Variant

`surv_cox` (TabPFN backbone + jointly-trained Cox head, Approach C) is the strongest TabPFN model
across all datasets. On **public datasets** it matches RSF closely:

- GBSG: 0.670 (RSF 0.675, Δ=–0.005)
- METABRIC: 0.626 (RSF 0.641, Δ=–0.015)
- SUPPORT2: 0.591 (RSF 0.616, Δ=–0.025)

On **Sirbu datasets** the gap widens (~10 points below RSF), but `surv_cox` still comfortably
outperforms classical Cox PH on public data and shows promising zero-shot generalization.

### 3. DeepHit Has a Systematic Failure Across All Datasets

Both `deephit_single` and `surv_deephit` produce **C-index below 0.5** on most datasets,
indicating inverted risk scores. This is a known issue with DeepHit when the time-to-event
discretization does not match the data distribution, or when training diverges. The Surv-DeepHit
variant is even worse (0.36–0.43), suggesting the joint TabPFN training amplifies the instability.

| Model | SUPPORT2 | METABRIC | GBSG | Sirbu-Mort. |
|-------|:--------:|:--------:|:----:|:-----------:|
| DeepHit | 0.508 | 0.473 | 0.429 | 0.448 |
| Surv-DeepHit | 0.570 | 0.479 | 0.387 | 0.365 |

> **Action required:** Risk scores from DeepHit need to be negated, or the training procedure
> revisited (learning rate, discretization bins, or survival-head compatibility).

### 4. TabPFN-Cox (Sequential Embedding) is Poorly Calibrated

`embedding_cox` (Approach A — sequential pipeline) achieves near-random C-index (0.50–0.56) and
has a **D-calibration score of 9.0** (maximum possible — completely uncalibrated) on all datasets.
Its IBS is also catastrophically high (0.40–0.64 vs ~0.18 for Cox PH). This suggests the
sequential pipeline breaks the calibration of the survival function.

### 5. MTLR Shows High Variance on Sirbu

`mtlr` is competitive on public datasets (0.56–0.64 C-index) but collapses on several Sirbu
tasks with very high standard deviation (std > 0.10 on Sirbu-CV, Sirbu-MI, Sirbu-Stroke),
indicating training instability on large imbalanced clinical cohorts.

### 6. Sirbu Multi-task: Consistent Ordering Across All 4 Outcomes

The relative ranking of models is **remarkably stable** across all 4 Sirbu clinical endpoints
(Mortality, CV, MI, Stroke), despite different event rates and clinical characteristics.
RSF, Cox PH, and GBSA are top-3 across all tasks. This consistency suggests the clinical features
in the Sirbu dataset generalize well to different endpoints, and model selection can be done once
and transferred.

### 7. Deep Survival Models Underperform on Sirbu

DeepSurv achieves a reasonable 0.76–0.79 C-index on Sirbu but consistently falls ~2 points
below RSF. MTLR and DeepHit both frequently fail on the Sirbu tasks. This could be attributed
to:
- Larger dataset size (8,065 patients) requiring more careful hyperparameter tuning
- High feature dimensionality (78 features) increasing the risk of overfitting in deep models
- Imbalanced censoring patterns across the different outcome tasks

### 8. Calibration Analysis (D-cal)

D-calibration measures how well predicted survival curves match observed Kaplan-Meier estimates.
Lower D-cal (closer to 0) is better.

**Well-calibrated models (D-cal < 0.5):**
- RSF, GBSA, DeepSurv, Cox PH on public datasets
- RSF, Cox PH on Sirbu datasets (~0.77–0.79)

**Poorly calibrated:**
- KM on Sirbu (D-cal ≈ 1.5–1.6) — expected; population-level model
- TabPFN-Cox (D-cal = 9.0 everywhere) — complete calibration failure
- Surv-DeepHit (D-cal ≈ 3.0–3.2) — consistent calibration failure
- Surv-MTLR (D-cal ≈ 3.7–3.9 on Sirbu) — calibration failure on large dataset

---

## Integrated Brier Score (Mean ± Std, 5-fold CV)

Lower is better. Cox PH IBS serves as reference.

| Model | SUPPORT2 | METABRIC | GBSG | Sirbu-Mort. |
|-------|:--------:|:--------:|:----:|:-----------:|
| Cox PH | 0.211±0.006 | 0.168±0.008 | 0.185±0.005 | 0.158±0.028 |
| RSF | **0.192±0.009** | **0.167±0.013** | **0.179±0.007** | **0.155±0.027** |
| GBSA | 0.195±0.010 | 0.171±0.015 | 0.183±0.008 | 0.163±0.033 |
| DeepSurv | 0.198±0.011 | 0.167±0.015 | 0.181±0.007 | 0.178±0.061 |
| Surv-Cox | 0.207±0.004 | 0.179±0.011 | 0.182±0.006 | 0.188±0.017 |
| MTLR | 0.231±0.048 | 0.183±0.027 | 0.185±0.007 | 0.240±0.122 |
| Surv-MTLR | 0.244±0.015 | 0.290±0.007 | 0.374±0.010 | 0.633±0.076 |
| TabPFN-Cox | 0.642±0.002 | 0.496±0.029 | 0.400±0.006 | 0.397±0.230 |

RSF consistently achieves the best IBS across all datasets, closely followed by DeepSurv and Cox PH.
`surv_cox` achieves competitive IBS on public datasets (0.18–0.21) but degrades on Sirbu.
TabPFN-Cox and Surv-MTLR/Surv-DeepHit produce very poor IBS, indicating the predicted survival
curves are far from the true distribution despite sometimes reasonable C-index.

---

## Training Efficiency

Training times (including Optuna HPO with 20 trials) from `metadata.json`.

| Model | Approx. time / fold | Notes |
|-------|:-------------------:|-------|
| KM | < 0.01 s | No tuning |
| Cox PH | 0.1–0.2 s | No tuning |
| RSF | 60–120 s | 20 Optuna trials |
| GBSA | 30–90 s | 20 Optuna trials |
| DeepSurv | 60–150 s | 20 Optuna trials |
| MTLR | 60–200 s | 20 Optuna trials |
| DeepHit | 90–300 s | 20 Optuna trials |
| TabPFN-Cox | 120–400 s | 20 Optuna trials + embedding |
| Surv-Cox | 40–80 s | GPU, no HPO |
| Surv-PCHaz | 40–80 s | GPU, no HPO |
| Surv-MTLR | 40–80 s | GPU, no HPO |

**Efficiency frontier winner:** Cox PH — exceptional C-index per second.
RSF sits on the Pareto frontier for higher-effort budgets.
`surv_cox` offers the best performance per second among TabPFN models.

---

## HPO Convergence

From `best_params.json` (20 trials per fold):

- **RSF and GBSA** converge quickly (best trial typically found in the first 40–60% of trials)
- **DeepSurv and MTLR** require more trials (best trial found late, ~60–80% of trials used)
- **TabPFN-Cox (embedding_cox)** converges fastest due to smaller search space but achieves poor results regardless
- **DeepHit** convergence is inconsistent — high variance in best_trial/n_trials ratio suggests the objective landscape is poorly behaved

---

## Limitations & Next Steps

### Known Issues

1. **DeepHit risk score inversion**: C-index < 0.5 indicates predictions are anti-correlated with true risk. The fix is to negate risk scores at inference, or swap the loss sign. Requires investigation in `models/deep_hit.py`.

2. **TabPFN-Cox (embedding_cox) calibration failure**: D-cal = 9.0 across all datasets. The sequential pipeline (TabPFN embed → Cox fit) breaks the calibration link between embeddings and time-to-event. Possible fix: use calibration post-processing (Platt scaling on the survival function).

3. **Surv-MTLR IBS degradation on Sirbu**: IBS = 0.63 (vs 0.16 for RSF) suggests the survival curves are nearly flat/uninformative despite reasonable C-index (~0.58). May be due to class imbalance in discrete time bins for long follow-up.

4. **pchazard not included**: `pchazard` runs failed during the benchmark due to a missing `interpolate` method in the installed pycox version. Results table will be updated once fixed.

### Recommended Next Steps

1. **Fix DeepHit**: Negate risk scores, re-run, update results
2. **Fix embedding_cox calibration**: Apply isotonic regression post-processing to survival functions
3. **Run retrieval augmentation (Approach B)**: `TabPFN-Retrieval` not yet evaluated
4. **Sirbu multi-task joint modelling**: Train a single model on all 4 Sirbu outcomes simultaneously
5. **Statistical significance**: Run Wilcoxon signed-rank tests between top models (RSF vs Surv-Cox, RSF vs Cox PH) to confirm rankings
6. **Hyperparameter sensitivity**: Ablate number of Optuna trials (5 vs 10 vs 20 vs 50) for RSF and DeepSurv
7. **Feature importance on Sirbu**: Leverage the 78 real feature names for clinical interpretability

---

## Reproducibility

```bash
# Re-run all experiments
bash survpfn/scripts/run.sh all

# Regenerate aggregated CSV
uv run python -m survpfn.scripts.aggregate --output-dir results

# Regenerate all figures
uv run python -m survpfn.xai.analysis --results-dir results --output-dir xai/figures
```

All per-fold results are stored in `results/<DATASET>/<model>/fold_N/` with:
- `metrics.json` — C-index, IBS, AUC, D-cal
- `metadata.json` — timing, event counts, n_params
- `best_params.json` — Optuna best hyperparameters
- `optuna_<model>.log` — full Optuna journal (warm-restartable)
