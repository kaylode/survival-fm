# SurvPFN

**In-Context Survival Analysis via Tabular Foundation Models**

[![Paper](https://img.shields.io/badge/paper-under%20review-yellow)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## Abstract

Survival analysis on clinical tabular data remains challenging: classical Cox proportional hazards and even modern deep survival models depend on handcrafted features and strong parametric assumptions that rarely hold in practice. SurvPFN proposes using **Tabular Foundation Models (TabPFN, TabDPT, TabICL)** as powerful embedding backbones for survival analysis. We evaluate three complementary strategies — frozen transfer, joint adaptation, and zero-shot in-context learning — with four survival heads (Cox, DeepHit, MTLR, PC-Hazard) on both single-risk and competing-risks settings.

---

## Key Ideas

```
Strategy 1 — Frozen Transfer  ({fm}_embedding_{head})
  X → FM Encoder (frozen) → Embedding [precomputed once] → Survival Head → S(t|x)

Strategy 2 — Joint Adaptation  ({fm}_joint_{head})
  X → FM Encoder (frozen weights, full forward per batch) → Survival Head → S(t|x)

Strategy 3 — Zero-shot ICL  ({fm}_zeroshot[_perbin])
  X → FM used directly as binary classifier per time bin → S(t|x)
  No head training. Implements Kim et al. (2026) ICL survival algorithm.

Competing Risks (CR) extension:
  Strategies 1 & 2 with DeepHit-CR head → CIF(t|x) per cause
  Strategy 3 with cause-specific FM classifiers → CIF_c ≈ 1 − S_c
```

---

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/<your-org>/survpfn.git
cd survpfn
uv sync
```

> **Note:** SurvPFN requires the full `tabpfn` package (local inference), **not** `tabpfn-client`. Resolved automatically by `uv sync`.

---

## Datasets

### Single-Risk Public Benchmarks
| Dataset | N | Source |
|---------|--:|--------|
| SUPPORT2 | ~9,000 | PyCox |
| METABRIC | ~1,900 | PyCox |
| GBSG | ~2,800 | PyCox |
| WHAS500 | 500 | sksurv |
| VETERANS | 137 | sksurv |
| FLCHAIN | 6,524 | sksurv |
| SEER | ~4,024 | Local CSV (`data/SEER Breast Cancer Dataset .csv`) |
| SS_* (25 datasets) | Var. | SurvSet |

### Competing-Risks Datasets
| Dataset | N | Causes |
|---------|--:|--------|
| METABRIC_CR | ~1,900 | single-event (CR models auto-detect n_risks=1) |
| SEER_CR | ~4,024 | single-event (CR models auto-detect n_risks=1) |

### EHR Datasets (private)
| Dataset | N | Notes |
|---------|--:|-------|
| EICU_SURV | ~80,000 | eICU Collaborative Research Database |
| MIMIC_SURV_B | ~40,000 | MIMIC-IV binary outcome cohort |

Public datasets download automatically on first use. SEER requires placing the CSV in `data/`. EHR datasets require credentialed access.

---

## Model Registry

All models follow the same call signature:

```python
from survpfn.models import ALL_MODELS

model, risk, surv_probs, time_grid = ALL_MODELS["tabicl_joint_deephit"](
    df_train, df_test, dur_col, ev_col,
    epochs=100, lr=1e-3, device="cuda:0",
)
```

### Single-Risk Models

| Group | Example registry keys | Count |
|-------|-----------------------|------:|
| Classical | `cox` `km` `rsf` `gbsa` | 4 |
| Deep | `deepsurv` `mtlr` `pchazard` `deephit_single` `survtrace` `soden` `beta_surv` | 7 |
| Frozen Transfer | `{tabpfn,tabdpt,tabicl}_embedding_{cox,deephit,pchazard,mtlr}` | 12 |
| Joint Adaptation | `{tabpfn,tabdpt,tabicl}_joint_{cox,deephit,pchazard,mtlr}` | 12 |
| Zero-shot ICL | `{tabpfn,tabdpt,tabicl}_zeroshot[_perbin]` | 6 |

### Competing-Risks Models

| Group | Example registry keys |
|-------|-----------------------|
| Classical CR | `aalen_johansen` `fine_gray` `rsf_cr` `deephit_cr` |
| FM Frozen CR | `{tabpfn,tabdpt,tabicl}_deephit_cr` |
| FM Zero-shot CR | `{tabpfn,tabdpt,tabicl}_zeroshot_cr` |

---

## Running Experiments

### Single-Risk Benchmark

```bash
# All models, all public datasets
bash survpfn/scripts/run_sr.sh all public

# One model group on a specific dataset
bash survpfn/scripts/run_sr.sh classical GBSG
bash survpfn/scripts/run_sr.sh fm_embedding "GBSG METABRIC"
bash survpfn/scripts/run_sr.sh zeroshot

# EHR datasets
bash survpfn/scripts/run_sr.sh classical ehr
```

**Groups:** `classical` · `deep` · `fm_embedding` · `fm_joint` · `tabpfn` · `tabdpt` · `tabicl` · `zeroshot` · `zeroshot_perbin` · `beta` · `all`

**Dataset keywords:** `public` · `survset` · `ehr` · (default: public)

### Competing-Risks Benchmark

```bash
bash survpfn/scripts/run_cr.sh all
bash survpfn/scripts/run_cr.sh classical_cr
bash survpfn/scripts/run_cr.sh zeroshot_cr METABRIC_CR
```

### Direct CLI

```bash
# Standard 5-fold CV
uv run python -m survpfn.scripts.benchmark \
    --datasets GBSG METABRIC \
    --models tabicl_embedding_cox rsf cox \
    --folds 5 --device cuda:0

# Label-efficiency sweep
uv run python -m survpfn.scripts.benchmark \
    --datasets SEER METABRIC \
    --models cox rsf tabicl_embedding_cox \
    --label-fractions 0.05 0.1 0.25 0.5 1.0

# Temporal split
uv run python -m survpfn.scripts.benchmark \
    --datasets SEER METABRIC \
    --models cox rsf tabicl_embedding_cox \
    --temporal-split --temporal-frac-train 0.70
```

### Analysis

```bash
# Aggregate per-fold JSONs → single CSV
uv run survpfn-aggregate --results-dir results/benchmark

# Statistical significance (Wilcoxon / Friedman / Nemenyi)
./survpfn/scripts/run_statistical.sh --all-metrics --references rsf cox

# LaTeX tables
uv run survpfn-latex-table --summary
```

---

## Evaluation Metrics

| Metric | SR | CR | Description |
|--------|:--:|:--:|-------------|
| C-Index | ✓ | ✓ | Harrell's / Antolini's concordance |
| IBS | ✓ | ✓ | Integrated Brier Score (IPCW-weighted) |
| AUROC(t) | ✓ | ✓ | Time-dependent AUC |
| D-Cal | ✓ | — | Haider's distributional calibration |

For CR: per-cause metrics are computed and macro-averaged as headline numbers.

---

## Project Structure

```
survpfn/
├── configs/              # Optuna HPO JSON per model
├── dataloaders/
│   ├── single.py         # Public single-risk loaders (PyCox, sksurv, SEER, SurvSet)
│   ├── competing.py      # Competing-risks dataset loaders
│   └── data_utils/       # Shared preprocessing helpers
├── metrics/              # C-index, IBS, AUC, D-cal, CR metrics, calibration
├── models/
│   ├── shared/
│   │   ├── preprocessing.py  # FMDataPrep, prepare_targets (shared across FMs)
│   │   ├── finetune.py       # BaseJointSurvFinetune, BaseSurvExpandedFinetune,
│   │   │                     # EventBalancedBatchSampler
│   │   ├── zeroshot.py       # ZeroShotSurvivalPredictor (Kim et al. 2026)
│   │   ├── binning.py        # Time discretisation utilities
│   │   └── loss.py           # Survival loss functions
│   ├── tabpfn/           # embedding.py · model.py (joint) · survival.py
│   ├── tabdpt/           # embedding.py · model.py (joint) · survival.py
│   ├── tabicl/           # embedding.py · model.py (joint) · survival.py
│   ├── sr_models/        # classical.py · tree.py · deep_surv.py · deep_hit.py
│   │                     # soden.py · beta_surv.py · survtrace/
│   ├── cr_models/        # competing_risks.py (AJ, Fine-Gray, RSF-CR, DeepHit-CR)
│   └── __init__.py       # ALL_MODELS registry
├── scripts/
│   ├── benchmark.py      # 5-fold CV runner (survpfn-benchmark)
│   ├── significance.py   # Wilcoxon / Friedman / Nemenyi / CD diagram
│   ├── latex_table.py    # Publication-ready LaTeX tables
│   ├── run_sr.sh         # Single-risk bulk runner
│   ├── run_cr.sh         # Competing-risks bulk runner
│   └── run_statistical.sh
└── xai/                  # Aggregation, plotting, explainability
tests/
```

---

## Contributing

1. Fork and create a feature branch.
2. Add dependencies with `uv add <package>`.
3. Follow ruff style: `uv run ruff check . && uv run ruff format .`
4. Add tests under `tests/` — run with `uv run pytest`.
5. Open a PR with a description.
