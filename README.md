# SurvPFN

**In-Context Survival Analysis via Tabular Foundation Models**

[![Paper](https://img.shields.io/badge/paper-under%20review-yellow)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## Abstract

Survival analysis on clinical tabular data remains challenging: classical Cox proportional hazards and even modern deep survival models depend on handcrafted features and strong parametric assumptions that rarely hold in practice. SurvPFN proposes using **Tabular Foundation Models (TabPFN, TabDPT, TabICL)** — Bayesian in-context learning models trained as prior-fitted networks — as powerful embedding backbones, decoupled from the choice of survival head. We further equip SurvPFN with:
1. **Retrieval Augmentation**: Scale in-context learning to large clinical cohorts (N > 1000) by selecting the most informative patient contexts.
2. **Competing Risks Support**: Dedicated pathways for medical scenarios with multiple competing event types.
3. **Modular Adapters**: Linear adapters to align foundation model representations with survival tasks.

We evaluate on several private cardiovascular datasets (Sirbu, URRAH), the **SurvSet Benchmark** (31+ healthcare datasets), and standard public benchmarks (SUPPORT2, METABRIC, GBSG).

---

## Key Ideas

```
Approach A — Foundation Embedding Backbone (SR & CR)
  Raw features → FM Encoder (TabPFN/TabDPT/TabICL) → [Adapter] → Embedding → Survival Head → S(t|x) or CIF(t|x)
  Heads available: Cox, DeepHit, MTLR, PC-Hazard, SurvTrace, DeepHit-CR, etc.

Approach B — Retrieval-Augmented ICL
  Raw features → Retrieval (top-K similar patients) → FM Encoder → Embedding → Survival Head → S(t|x)

Approach C — Joint End-to-End Fine-Tuning
  Raw features → FM Encoder ──┐
                    (fine-tuned)  ├─ Joint survival loss → S(t|x) / CIF(t|x)
  Survival Head ─────────────────┘
```


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

| Source | Dataset | N | Features | Outcomes | Access |
|--------|---------|--:|---------|----------|--------|
| **Sirbu** | `SIRBU_mortality` | 8,065 | 78 | All-cause mortality | Restricted |
| | `SIRBU_cv` | 8,065 | 78 | Cardiovascular mortality | Restricted |
| | `SIRBU_mi` | 8,065 | 78 | Myocardial infarction | Restricted |
| | `SIRBU_stroke` | 8,065 | 78 | Stroke | Restricted |
| **PyCox** | `SUPPORT2` | ~9,000 | 14 | In-hospital mortality | Public |
| | `METABRIC` | ~1,900 | 9 | Breast cancer survival | Public |
| | `GBSG` | ~2,800 | 7 | Breast cancer recurrence | Public |
| **SurvSet** | `SURVSET_BENCHMARK` | 31+ | Var. | Diverse healthcare | Public |

Public datasets are downloaded automatically. Private datasets require Excel source files in `Dataset Sirbu/`.


---

## Models

### Foundation Model Adapters
| Category | Strategy | Models |
|----------|----------|--------|
| **Embedding** | Frozen backbone + head | `tabpfn_embedding_*`, `tabdpt_embedding_*`, `tabicl_embedding_*` |
| **Joint** | Fine-tuned backbone + head | `tabpfn_joint_*`, `tabdpt_joint_*`, `tabicl_joint_*` |
| **Zero-Shot** | Direct ICL (no head) | `tabpfn_zeroshot`, `tabdpt_zeroshot`, `tabicl_zeroshot` |

*Available heads: `cox`, `deephit`, `pchazard`, `mtlr`. Append `_adapter` for learnable linear projection.*

### Classical & Deep Learning (Single Risk)
| Family | Models |
|--------|--------|
| **Classical** | `km` (Kaplan-Meier), `cox` (Cox PH) |
| **Tree-based** | `rsf` (Random Survival Forest), `gbsa` (Gradient Boosting) |
| **Neural** | `deepsurv`, `deephit_single`, `mtlr`, `pchazard`, `survtrace`, `soden`, `beta_surv` |

### Competing Risks (CR)
| Family | Models |
|--------|--------|
| **Classical** | `cox_cr` (Fine-Gray / Cause-specific) |
| **Neural** | `deephit_cr` |
| **Foundation** | `tabpfn_embedding_deephit_cr`, `tabpfn_joint_deephit_cr`, `tabpfn_zeroshot_cr_*` |

---

## Evaluation Metrics

| Metric | Task | Description |
|--------|------|-------------|
| **C-Index** | SR & CR | Harrell's / Antolini's concordance index |
| **IBS** | SR & CR | Integrated Brier Score (calibration + discrimination) |
| **AUROC(t)** | SR & CR | Time-dependent Area Under the ROC Curve |
| **D-Cal** | SR | Haider's Distributional Calibration |


---

## Running Experiments

### Quick Start — Single Risk (SR)
```bash
# Individual model run (e.g., GBSG dataset with RSF)
uv run survpfn/scripts/benchmark.py --datasets GBSG --models rsf --tune --trials 20

# Bulk run (all datasets and SR models)
bash survpfn/scripts/run_sr.sh all
```

### Quick Start — Competing Risks (CR)
```bash
# Foundation model CR run (SIRBU Myocardial Infarction task)
uv run survpfn/scripts/benchmark.py \
    --datasets SIRBU_mi \
    --models tabpfn_embedding_deephit_cr \
    --device cuda:0

# Bulk run (all datasets and CR models)
bash survpfn/scripts/run_cr.sh all
```

### Analysis & Reporting
After experiments complete, aggregate results and generate publication-ready tables:

```bash
# 1. Aggregate results from /results folder
uv run survpfn/scripts/aggregate.py --results-dir results

# 2. Generate LaTeX tables (C-index, IBS)
uv run survpfn/scripts/latex_table.py --input results/aggregated.csv

# 3. Statistical Significance (Wilcoxon signed-rank tests)
uv run survpfn/scripts/significance.py --input results/aggregated.csv
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
├── dataloaders/                    # Dataset loaders (SurvSet, Sirbu, PyCox)
├── metrics/                        # Evaluation metrics
│   ├── single.py                   # Single-risk (C-index, IBS, AUROC)
│   ├── competing.py                # Competing-risks (CIF-based metrics)
│   └── calibration.py              # D-Calibration (Haider 2020)
├── models/
│   ├── sr_models/                  # Single-risk: Cox, RSF, DeepSurv, Soden...
│   ├── cr_models/                  # Competing-risks: CR-Cox, DeepHit-CR
│   ├── tabpfn/                     # TabPFN adapters (Embedding, Joint, Zero-Shot)
│   ├── tabdpt/                     # TabDPT adapters (Embedding, Joint, Zero-Shot)
│   └── tabicl/                     # TabICL adapters (Embedding, Joint, Zero-Shot)
├── scripts/
│   ├── benchmark.py                # Unified CV runner (entry point)
│   ├── run_sr.sh / run_cr.sh       # Bulk experiment launchers
│   ├── latex_table.py              # Generate publication results
│   └── significance.py             # Statistical tests (Wilcoxon)
└── xai/                            # Explainability & plotting
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
