# SurvPFN

**In-Context Survival Analysis via Tabular Foundation Models with Retrieval Augmentation**

[![Paper](https://img.shields.io/badge/paper-under%20review-yellow)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## Abstract

Survival analysis on clinical tabular data remains challenging: classical Cox proportional hazards and even modern deep survival models depend on handcrafted features and strong parametric assumptions that rarely hold in practice. SurvPFN proposes using TabPFN — a Bayesian in-context learning model trained as a prior-fitted network — as a powerful embedding backbone, decoupled from the choice of survival head. We further equip SurvPFN with retrieval augmentation to scale in-context learning to large clinical cohorts (N > 1000) by selecting the most informative patient contexts. We evaluate on two private cardiovascular datasets (Sirbu, URRAH) and three public benchmarks (SUPPORT2, METABRIC, GBSG).

---

## Key Ideas

```
Approach A — Primary (TabPFN-Cox / TabPFN-Surv)
  Raw features → TabPFN Encoder → Embedding → Survival Head → S(t|x)
  Heads available: Cox, DeepHit, MTLR, PC-Hazard

Approach B — Secondary (TabPFN-Retrieval)
  Raw features → Retrieval (top-K similar patients) → TabPFN Encoder → Embedding → Cox → S(t|x)

Approach C — Future (End-to-End Fine-Tuning)
  Raw features → TabPFN Encoder ──┐
                    (fine-tuned)  ├─ Joint survival loss → S(t|x)
  Survival Head ─────────────────┘
```

The survival head is modular: any head (Cox, DeepHit, MTLR, PC-Hazard) can be plugged in on top of the TabPFN embedding, enabling a clean embedding × head ablation grid.

---

## Installation

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/<your-org>/survpfn.git
cd survpfn
uv sync
```

> **Note:** SurvPFN requires the full `tabpfn` package (local inference), **not** `tabpfn-client`. The dependency is pinned in `pyproject.toml` and resolved automatically by `uv sync`.

---

## Datasets

| Dataset | N | Features | Outcomes | Access |
|---------|--:|---------|----------|--------|
| Sirbu — Total Mortality (`SIRBU_mortality`) | 8,065 | 78 | All-cause mortality | Restricted |
| Sirbu — CV Mortality (`SIRBU_cv`) | 8,065 | 78 | Cardiovascular mortality | Restricted |
| Sirbu — MI (`SIRBU_mi`) | 8,065 | 78 | Myocardial infarction | Restricted |
| Sirbu — Stroke (`SIRBU_stroke`) | 8,065 | 78 | Stroke | Restricted |
| SUPPORT2 | ~9,000 | 14 | In-hospital mortality | Public (pycox) |
| METABRIC | ~1,900 | 9 | Breast cancer survival | Public (pycox) |
| GBSG | ~2,800 | 7 | Breast cancer recurrence | Public (pycox) |

Public datasets are downloaded automatically via `pycox` on first use. Private datasets require institutional data access agreements and Excel source files in `Dataset Sirbu/`.

---

## Models

| Key | Model | Type | File |
|-----|-------|------|------|
| `km` | Kaplan-Meier | Classical baseline | `models/classical.py` |
| `cox` | Cox PH | Classical | `models/classical.py` |
| `rsf` | Random Survival Forest | Tree | `models/tree.py` |
| `gbsa` | Gradient Boosting Survival | Tree | `models/tree.py` |
| `deepsurv` | DeepSurv | Deep | `models/deep_surv.py` |
| `mtlr` | MTLR | Deep | `models/deep_hit.py` |
| `pchazard` | PC-Hazard | Deep | `models/deep_hit.py` |
| `deephit_single` | DeepHit (single risk) | Deep | `models/deep_hit.py` |
| `embedding_cox` | **TabPFN-Cox** | Foundation — Approach A | `models/tabpfn/cox.py` |
| `surv_cox` | **TabPFN-Surv (Cox head)** | Foundation — Approach C | `models/tabpfn/cox.py` |
| `surv_deephit` | **TabPFN-Surv (DeepHit head)** | Foundation — Approach C | `models/tabpfn/cox.py` |
| `surv_pchazard` | **TabPFN-Surv (PC-Hazard head)** | Foundation — Approach C | `models/tabpfn/cox.py` |
| `surv_mtlr` | **TabPFN-Surv (MTLR head)** | Foundation — Approach C | `models/tabpfn/cox.py` |

All tunable models (rsf, gbsa, deepsurv, mtlr, pchazard, deephit_single, embedding_cox) use **Optuna** for hyperparameter search with journal-based storage, enabling warm restarts across runs.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Harrell's C-index | Discrimination: probability of correct pair ordering |
| Integrated Brier Score (IBS) | Joint calibration + discrimination over time |
| Time-dependent AUROC | AUC at each evaluation time point |
| D-calibration (Haider 2020) | Distributional calibration of predicted survival curves |

---

## Running Experiments

### Quick start — one dataset, all models

```bash
uv run survpfn/scripts/benchmark.py \
    --datasets GBSG \
    --models all \
    --tune --trials 20 \
    --folds 5
```

### Bulk run — all datasets and model groups (recommended)

```bash
# Classical + tree models (CPU-friendly)
bash survpfn/scripts/run.sh classical

# Deep survival models
bash survpfn/scripts/run.sh deep

# TabPFN jointly-trained models (GPU recommended)
bash survpfn/scripts/run.sh tabpfn

# Everything at once
bash survpfn/scripts/run.sh all
```

Logs are written to `logs/run_<group>_<timestamp>.log`.

### Sirbu multi-task (4 outcomes)

```bash
uv run survpfn/scripts/benchmark.py \
    --datasets SIRBU_mortality SIRBU_cv SIRBU_mi SIRBU_stroke \
    --models all \
    --tune --trials 20 \
    --folds 5
```

### TabPFN jointly-trained models (GPU options)

```bash
uv run survpfn/scripts/benchmark.py \
    --datasets SUPPORT2 METABRIC GBSG SIRBU_mortality \
    --models surv_cox surv_deephit surv_pchazard surv_mtlr \
    --epochs 50 --lr 1e-3 --device cuda:0 \
    --folds 5
```

### Aggregate results into a single CSV

```bash
uv run survpfn/scripts/aggregate.py \
    --results-dir results \
    --output-dir results
# Writes: results/aggregated.csv  results/summary.csv
```

### Generate comparison figures

```bash
uv run survpfn/xai/plot_comparison.py \
    --aggregated results/aggregated.csv \
    --results-dir results \
    --output-dir xai/figures
```

---

## Results Structure

Each run writes an independent folder per (dataset, model, fold):

```
results/
  <DATASET>/                        # e.g. GBSG, SUPPORT2, SIRBU_mortality
    <model>/                        # e.g. rsf, cox, surv_cox
      fold_1/
        metrics.json                # C-index, IBS, AUC, D-cal
        metadata.json               # timing, event counts, n_params, config
        feature_importance.json     # Cox coefs / tree importances (if available)
        best_params.json            # Optuna best HPs + convergence info (if --tune)
        optuna_<model>.log          # Optuna journal for warm restart
      fold_2/ … fold_5/
  old/                              # Archived flat CSVs from earlier runs
```

No aggregation happens during training. Run `aggregate.py` after all experiments complete.

---

## Project Structure

```
survpfn/                            # Python package
├── dataloaders/
│   ├── dataloader.py               # Public loaders (SUPPORT2, METABRIC, GBSG)
│   │                               # + Sirbu multi-task loaders
│   └── preprocessing.py            # Sirbu-specific cleaning, imputation, task prep
├── metrics/
│   ├── metrics.py                  # C-index, IBS, time-dep AUC, D-calibration
│   └── plotting.py                 # Survival curve plotting utilities
├── models/
│   ├── classical.py                # Cox PH (lifelines)
│   ├── tree.py                     # RSF, GBSA (scikit-survival + Optuna)
│   ├── deep_surv.py                # DeepSurv (pycox + Optuna)
│   ├── deep_hit.py                 # DeepHit, MTLR, PC-Hazard (pycox + Optuna)
│   ├── custom.py                   # TorchSurv custom model
│   └── tabpfn/
│       ├── cox.py                  # TabPFN-Cox (Approach A) + TabPFN-Surv (Approach C)
│       ├── embedding.py            # TabPFN embedding extraction
│       ├── retrieval.py            # Retrieval augmentation (Approach B)
│       └── backbone/               # TabPFN transformer backbone utilities
├── scripts/
│   ├── benchmark.py                # Unified CV runner — all datasets & models
│   ├── aggregate.py                # Aggregate result folders → CSV + summary
│   └── run.sh                      # Bulk experiment launcher (all datasets/models)
├── utils/
│   └── io.py                       # I/O helpers
└── xai/
    └── plot_comparison.py          # Model comparison figures from aggregated CSV

notebooks/                          # Exploratory analysis
results/                            # Per-run output (see Results Structure above)
docs/                               # Research plan, literature survey, code review
Dataset Sirbu/                      # Private Sirbu source Excel files (restricted)
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add new dependencies with `uv add <package>` (updates `pyproject.toml` and `uv.lock`).
3. Follow the code style enforced by `ruff`:
   ```bash
   uv run ruff check .
   uv run ruff format .
   ```
4. Add tests under `tests/` — the project uses `pytest`.
5. Open a pull request with a clear description of the change.

---

## Citation

If you use SurvPFN in your research, please cite:

```bibtex
@article{survpfn2026,
  title     = {SurvPFN: In-Context Survival Analysis via Tabular Foundation Models
               with Retrieval Augmentation},
  author    = {...},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
