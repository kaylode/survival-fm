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
Approach A — Primary (TabPFN-Cox)
  Raw features → TabPFN Encoder → Embedding → Survival Head → S(t|x)

Approach B — Secondary (TabPFN-Retrieval)
  Raw features → Retrieval (top-K similar patients) → TabPFN Encoder → Embedding → Survival Head → S(t|x)

Approach C — Future (End-to-End Fine-Tuning)
  Raw features → TabPFN Encoder ──┐
                    (fine-tuned)  ├─ Joint survival loss → S(t|x)
  Survival Head ─────────────────┘
```

The survival head is modular: any head (Cox, DeepHit, MTLR, PC-Hazard, …) can be plugged in on top of the TabPFN embedding, enabling a clean 6 × 6 embedding × head ablation grid.

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
| Sirbu (private) | 8,065 | 78 | Total mortality, CVD, MI, Stroke | Restricted |
| URRAH (private) | 27,078 | 85 | Total mortality, CVD, MI, Stroke | Restricted |
| SUPPORT2 | ~9,000 | 14 | In-hospital mortality | Public (pycox) |
| METABRIC | ~1,900 | 9 | Breast cancer survival | Public (pycox) |
| GBSG | ~2,800 | 7 | Breast cancer recurrence | Public (pycox) |

Public datasets are downloaded automatically via `pycox` on first use. Private datasets require institutional data access agreements.

---

## Running Experiments

### Full benchmark (private data)

```bash
uv run python scripts/train.py --folds 5 --tune
```

### Public benchmark

```bash
uv run python scripts/benchmark.py --datasets SUPPORT2 METABRIC GBSG --folds 5
```

### TabPFN combination grid

```bash
# Test all 6 embedding × 6 survival head combinations
uv run python experiments/tabpfn_combos.py --dataset GBSG --folds 5

# With retrieval augmentation
uv run python experiments/tabpfn_combos.py --dataset sirbu \
    --embeddings tabpfn_retrieval_k10 tabpfn_retrieval_k50 \
    --heads cox deephit pchazard
```

Results are written to `results/` as CSV files and summary plots.

---

## Models

| Model | Type | File | Notes |
|-------|------|------|-------|
| Kaplan-Meier | Classical | `models/classical.py` | Population-level baseline |
| Cox PH | Classical | `models/classical.py` | Lifelines implementation |
| RSF | Tree | `models/tree.py` | Random Survival Forest |
| GBSA | Tree | `models/tree.py` | Gradient Boosting Survival Analysis |
| DeepSurv | Deep | `models/deep_surv.py` | Cox-based neural network |
| DeepHit | Deep | `models/deep_hit.py` | Discrete-time, competing risks |
| MTLR | Deep | `models/deep_hit.py` | Multi-task logistic regression |
| PC-Hazard | Deep | `models/deep_hit.py` | Piecewise-constant hazard |
| TorchSurv | Deep | `models/custom.py` | Custom TorchSurv model |
| **TabPFN-Cox** | Foundation | `models/tabpfn/cox.py` | **Our method (Approach A)** |
| **TabPFN-Retrieval** | Foundation | `models/tabpfn/retrieval.py` | **Our method (Approach B)** |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Harrell's C-index | Discrimination: probability of correct pair ordering |
| Integrated Brier Score (IBS) | Joint calibration + discrimination over time |
| Time-dependent AUROC | AUC at each evaluation time point |
| D-calibration (Haider 2020) | Distributional calibration of predicted survival curves |
| Wilcoxon signed-rank test | Statistical significance vs. Cox PH baseline (5-fold CV) |

---

## Results

> Full results are in progress. The table below will be updated upon completion of all experiments.

| Model | SUPPORT2 C-idx | METABRIC C-idx | GBSG C-idx | Sirbu C-idx |
|-------|:--------------:|:--------------:|:----------:|:-----------:|
| Cox PH | — | — | — | 0.788 ± 0.01 |
| RSF | — | — | — | 0.793 ± 0.01 |
| DeepSurv | — | — | — | 0.773 ± 0.02 |
| DeepHit | — | — | — | — |
| **TabPFN-Cox (ours)** | — | — | — | — |
| **TabPFN-Retrieval (ours)** | — | — | — | — |

---

## Project Structure

```
survpfn/                   # Python package
├── data/                  # Loading, preprocessing, benchmark loaders
├── models/
│   ├── classical.py       # Cox PH, Kaplan-Meier
│   ├── tree.py            # RSF, GBSA
│   ├── deep_surv.py       # DeepSurv
│   ├── deep_hit.py        # DeepHit, MTLR, PC-Hazard
│   ├── custom.py          # TorchSurv custom model
│   └── tabpfn/            # TabPFN embedding backbone + retrieval
├── eval/                  # Metrics (C-index, IBS, D-cal), plotting
└── utils/                 # I/O helpers

scripts/
├── train.py               # Full benchmark on private datasets
└── benchmark.py           # Benchmark on public datasets

experiments/
└── tabpfn_combos.py       # 6×6 embedding × survival head grid

notebooks/                 # Exploratory analysis
results/                   # Experiment outputs (CSV + plots)
tests/                     # Unit and integration tests
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
