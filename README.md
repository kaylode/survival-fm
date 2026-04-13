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
  X → FM Encoder (frozen weights, full forward per batch) → Head
  Loss = L_survival + α · L_classification   (backbone logits as aux signal)

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

## Configuration

All FM hyperparameters are managed through a single `FMConfig` dataclass:

```python
from survpfn.utils import FMConfig

cfg = FMConfig(
    head_type="deephit",      # cox | deephit | pchazard | mtlr
    num_durations=10,          # time bins for discrete heads
    context_size=256,          # ICL context samples per batch
    epochs=100,
    learning_rate=1e-3,
    alpha=1.0,                 # weight of FM classification auxiliary loss
    device="cuda:0",
)
# Also accepted as flat kwargs by any registered model:
# ALL_MODELS["tabicl_joint_deephit"](df_tr, df_ts, dur, ev, epochs=100, lr=1e-3)
```

See `survpfn/utils/config.py` for the full field list.

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
| SEER | ~4,024 | Local CSV |
| SS_* (26 datasets) | Var. | SurvSet |

### Competing-Risks Datasets
| Dataset | N | Causes |
|---------|--:|--------|
| FRAMINGHAM | ~4,400 | CVD / other-cause death |
| PBC2 | ~312 | death / liver transplant |
| SUPPORT_CR | ~9,000 | cancer death / non-cancer death |
| SYNTHETIC_CR | ~1,000 | synthetic 2-cause dataset |

Public datasets download automatically. SEER requires `data/SEER Breast Cancer Dataset .csv`.

---

## Model Registry

All models are registered in `survpfn/models/__init__.py` and follow the same call signature:

```python
from survpfn.models import ALL_MODELS

model, risk, surv_probs, time_grid = ALL_MODELS["tabicl_joint_deephit"](
    df_train, df_test, dur_col, ev_col,
    epochs=100, lr=1e-3, device="cuda:0",
)
```

### Single-Risk (SR) Models

| Group | Registry keys | Count |
|-------|--------------|------:|
| Classical | `cox` `km` `rsf` `gbsa` | 4 |
| Deep | `deepsurv` `mtlr` `pchazard` `deephit_single` `survtrace` `soden` `beta_surv` | 7 |
| Frozen Transfer | `{tabpfn,tabdpt,tabicl}_embedding_{cox,deepsurv,deephit,pchazard,mtlr}` | 15 |
| Joint Adaptation | `{tabpfn,tabdpt,tabicl}_joint_{cox,deepsurv,deephit,pchazard,mtlr}` | 15 |
| Zero-shot | `{tabpfn,tabdpt,tabicl}_zeroshot[_perbin]` | 6 |
| SurvAdapter | `{tabpfn,tabdpt,tabicl}_surv_adapter` | 3 |

### Competing-Risks (CR) Models

| Group | Registry keys | Count |
|-------|--------------|------:|
| Classical CR | `cox_cr` `aj_cr` `fine_gray_cr` `survival_boost_cr` | 4 |
| Deep CR | `deephit_cr` | 1 |
| Frozen Transfer CR | `{tabpfn,tabdpt,tabicl}_embedding_deephit_cr` | 3 |
| Joint Adaptation CR | `{tabpfn,tabdpt,tabicl}_joint_deephit_cr` | 3 |
| Zero-shot CR | `{tabpfn,tabdpt,tabicl}_zeroshot_cr` | 3 |

---

## Running Experiments

### Single-Risk Benchmark (`run_sr.sh`)

```bash
# All models, all public datasets
bash survpfn/scripts/run_sr.sh all public

# One model group
bash survpfn/scripts/run_sr.sh classical
bash survpfn/scripts/run_sr.sh fm_embedding
bash survpfn/scripts/run_sr.sh fm_joint
bash survpfn/scripts/run_sr.sh tabicl

# Specific dataset(s)
bash survpfn/scripts/run_sr.sh deep "GBSG METABRIC"

# Parallel mode (one background job per dataset)
bash survpfn/scripts/run_sr.sh all public --parallel
```

**Groups:** `classical` · `deep` · `fm` · `fm_embedding` · `fm_joint` · `tabpfn` · `tabdpt` · `tabicl` · `tabpfn_embedding` · `tabdpt_embedding` · `tabicl_embedding` · `tabpfn_joint` · `tabdpt_joint` · `tabicl_joint` · `zeroshot` · `zeroshot_perbin` · `surv_adapter` · `all`

**Dataset keywords:** `public` · `survset` · `ormoni_tirodei` · (default: public + survset)

### Competing-Risks Benchmark (`run_cr.sh`)

```bash
bash survpfn/scripts/run_cr.sh all
bash survpfn/scripts/run_cr.sh classical_cr
bash survpfn/scripts/run_cr.sh tabicl_cr
bash survpfn/scripts/run_cr.sh fm_joint FRAMINGHAM
```

**Groups:** `classical_cr` · `deep_cr` · `fm_embedding` · `fm_joint` · `fm_zeroshot` · `tabpfn_cr` · `tabdpt_cr` · `tabicl_cr` · `all`

### Direct CLI (benchmark.py)

```bash
# Single run with tuning
uv run python -m survpfn.scripts.benchmark \
    --datasets GBSG METABRIC \
    --models tabicl_embedding_cox rsf cox \
    --folds 5 --tune --trials 20 \
    --device cuda:0

# Label-efficiency experiment
uv run python -m survpfn.scripts.benchmark \
    --datasets SEER METABRIC \
    --models tabicl_embedding_cox rsf cox \
    --label-fractions 0.05 0.1 0.25 0.5 1.0

# Temporal (prospective) split — row order treated as enrollment order
uv run python -m survpfn.scripts.benchmark \
    --datasets SEER METABRIC \
    --models cox rsf tabicl_embedding_cox \
    --temporal-split --temporal-frac-train 0.70

# All FM models, one dataset
uv run python -m survpfn.scripts.benchmark \
    --datasets GBSG --models fm --epochs 200 --lr 1e-3
```

### Analysis

```bash
# Aggregate per-fold JSONs → single CSV
uv run survpfn-aggregate --results-dir results/benchmark

# Statistical significance (Wilcoxon / Friedman / Nemenyi)
./run_statistical.sh --all-metrics --references rsf cox

# LaTeX tables
uv run survpfn-latex-table --summary
```

---

## Data Preprocessing Pipeline

All three FM joint models (TabPFN, TabICL, TabDPT) share the same preprocessing pipeline, implemented in `survpfn/models/shared/preprocessing.py`:

```
Raw features (N, F)
  │
  ├─ StandardScaler.fit_transform()       → zero mean, unit variance
  │
  ├─ [Optional] PCA to backbone capacity  → when F > max_features
  │   TabPFN: max_features = 2000
  │   TabDPT: max_features = model.num_features (from checkpoint)
  │   TabICL: handles variable F internally — no PCA needed
  │
  └─ float32 tensor on target device
```

`FMDataPrep` handles steps 1–2. `prepare_targets` handles label transforms (Cox / discrete / CR) consistently across all three models.

---

## Evaluation Metrics

| Metric | SR | CR | Description |
|--------|:--:|:--:|-------------|
| C-Index | ✓ | ✓ | Harrell's / Antolini's concordance |
| IBS | ✓ | ✓ | Integrated Brier Score (IPCW-weighted) |
| AUROC(t) | ✓ | ✓ | Time-dependent AUC |
| D-Cal | ✓ | — | Haider's distributional calibration |

For CR: per-cause metrics are computed then macro-averaged as headline numbers.

---

## Tests

```bash
# Core FM tests (no checkpoint required for TabICL/TabPFN — auto-downloads)
uv run pytest tests/test_fm_embeddings.py -v   # frozen embedding extraction + LOO
uv run pytest tests/test_fm_joint.py -v        # fit/predict, loss shapes, float32 checks
uv run pytest tests/test_fm_zeroshot.py -v     # zero-shot ICL shapes + monotonicity

# TabDPT tests require checkpoint
TABDPT_CHECKPOINT=/path/to/tabdpt1_1.pth uv run pytest tests/ -v

# Dataloaders + metrics
uv run pytest tests/test_dataloaders.py tests/test_metrics.py -v
```

---

## Results Structure

```
results/benchmark/
  <DATASET>/
    <model>/
      fold_1/
        metrics.json          # C-index, IBS, AUC, D-cal
        metadata.json         # timing, n_params, event counts, config
        feature_importance.json
        best_params.json      # Optuna best HPs (when --tune)
      fold_2/ … fold_5/
results/benchmark_cr/         # Competing-risks results (separate output dir)
```

Run `uv run survpfn-aggregate` after all experiments to collect results into a single CSV.

---

## Project Structure

```
survpfn/
├── configs/              # Optuna HPO JSON per model
├── dataloaders/          # Dataset loaders (PyCox, SurvSet, SEER, OrmoniTirodei)
├── metrics/              # C-index, IBS, AUC, D-cal, CR metrics, calibration
├── models/
│   ├── shared/
│   │   ├── preprocessing.py   # FMDataPrep, prepare_targets (shared across FMs)
│   │   ├── heads.py           # Shared survival head builders + CR loss functions
│   │   └── zeroshot_surv.py   # Zero-shot ICL core algorithm
│   ├── tabpfn/            # TabPFN: embedding.py, model.py (joint), survival.py
│   ├── tabdpt/            # TabDPT: embedding.py, model.py (joint), survival.py
│   ├── tabicl/            # TabICL: embedding.py, model.py (joint), survival.py
│   ├── sr_models/         # Cox, RSF, GBSA, DeepSurv, DeepHit, SurvTRACE, SODEN, BetaSurv
│   ├── cr_models/         # AJ, Fine-Gray, SurvivalBoost-CR, DeepHit-CR
│   └── __init__.py        # ALL_MODELS registry (~56 models, 3 wrapper factories)
├── scripts/
│   ├── benchmark.py       # Unified CV runner (CLI entry point)
│   ├── _lib.sh            # Shared shell helpers (sourced by run_sr/run_cr)
│   ├── run_sr.sh          # Single-risk bulk runner
│   ├── run_cr.sh          # Competing-risks bulk runner
│   ├── run_statistical.sh # Statistical significance testing
│   ├── significance.py    # Wilcoxon / Friedman / Nemenyi / CD diagram
│   └── latex_table.py     # Publication-ready LaTeX tables
├── utils/
│   ├── config.py          # FMConfig dataclass (centralized hyperparameters)
│   └── ...
└── xai/                   # Aggregation, plotting, explainability
tests/
├── test_dataloaders.py
├── test_fm_embeddings.py  # Embedding extraction + LOO correctness
├── test_fm_joint.py       # Joint adaptation fit/predict/loss
└── test_fm_zeroshot.py    # Zero-shot ICL shapes + monotonicity
docs/
└── REFACTORING.md         # Refactoring sprint notes (2026-04-08)
```

---

## Contributing

1. Fork and create a feature branch.
2. Add dependencies with `uv add <package>`.
3. Follow ruff style: `uv run ruff check . && uv run ruff format .`
4. Add tests under `tests/` — run with `uv run pytest`.
5. Open a PR with a description.
