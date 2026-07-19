# SurvFM: Survival-Aware Adaptation of Tabular Foundation Models

[![Paper](https://img.shields.io/badge/paper-AIiH%202026-green)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

SurvFM is a modular framework for adapting tabular foundation models to survival analysis. It connects pretrained tabular backbones such as TabPFN, TabDPT, and TabICL with censoring-aware survival heads, including Cox proportional hazards, DeepHit, MTLR, and PC-Hazard, and evaluates them against classical, tree-based, and deep survival baselines.

The project studies whether tabular foundation model priors transfer to clinical time-to-event prediction, where labels are right-censored and the target is a survival function rather than a class probability. SurvFM proposes multiple adaptation strategies: frozen-backbone survival heads, jointly adapted backbone-head models, temporal expansion fine-tuning, and zero-shot in-context survival prediction through time-bin discretisation.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <repo-url>
cd survpfn
uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

Some foundation-model checkpoints are stored under `survpfn/models/models_diff/` or downloaded by the corresponding backend package on first use.

## Quick Start

The model registry in `survpfn.models.ALL_MODELS` exposes the training entry points used by the benchmark. Each callable expects a training dataframe, a test dataframe, the duration column name, and the event column name. It returns:

```python
model, risk_scores, survival_probabilities, time_grid
```

Example: fine-tune a TabPFN-based survival model with a Cox head.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from survpfn.models import ALL_MODELS

# Example data: feature columns plus duration and event columns.
# Replace this with your own dataframe.
rng = np.random.default_rng(42)
df = pd.DataFrame(
    {
        "age": rng.normal(65, 10, 200),
        "marker": rng.normal(0, 1, 200),
        "duration": rng.uniform(1, 100, 200),
        "event": rng.integers(0, 2, 200),
    }
)

df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["event"],
)

train_tabpfn_cox = ALL_MODELS["tabpfn_embedding_cox"]

model, risk, surv_probs, times = train_tabpfn_cox(
    df_train,
    df_test,
    "duration",
    "event",
    device="cuda:0",      # use "cpu" if CUDA is unavailable
    num_durations=10,
    epochs=20,
    batch_size=32,
    random_state=42,
    tune=False,
)

print(risk.shape)        # one risk score per test row
print(surv_probs.shape)  # rows=test samples, columns=time grid
print(times[:5])
```

For lower-level control, instantiate the TabPFN survival wrapper directly:

```python
import numpy as np

from survpfn.models.tabpfn import TabPFNSurvPH

feature_cols = [c for c in df_train.columns if c not in {"duration", "event"}]

model = TabPFNSurvPH(
    head_type="cox",
    freeze_tabpfn=True,
    num_durations=10,
    input_dim=len(feature_cols),
    context_size=256,
    device="cuda:0",
    epochs=20,
    batch_size=32,
)

model.fit(
    df_train[feature_cols].to_numpy(dtype=np.float32),
    df_train["duration"].to_numpy(),
    df_train["event"].to_numpy(),
)

survival_df = model.predict_survival_df(
    df_test[feature_cols].to_numpy(dtype=np.float32)
)
```

## Project Structure

```text
survpfn/
|-- README.md                  # Project overview and usage notes
|-- pyproject.toml             # Package metadata and dependencies
|-- uv.lock                    # Reproducible uv dependency lockfile
|-- docs/                      # Additional project notes and documentation
|-- checkpoints/               # Local model checkpoints
|-- data -> ...                # Local dataset 
|-- results -> ...             # Local benchmark output 
|-- survpfn/
|   |-- configs/               # Model and tuning configuration JSON files
|   |-- dataloaders/           # Public, EHR, SurvSet, and competing-risk loaders
|   |-- metrics/               # Survival, competing-risk, calibration, and stats metrics
|   |-- models/                # Classical baselines, deep survival models, and FM adapters
|   |   |-- tabpfn/            # TabPFN backbone plus survival wrappers
|   |   |-- tabdpt/            # TabDPT backbone plus survival wrappers
|   |   |-- tabicl/            # TabICL backbone plus survival wrappers
|   |   |-- shared/            # Shared fine-tuning, losses, preprocessing, and binning
|   |   |-- sr_models/         # Single-risk survival baselines
|   |   `-- cr_models/         # Competing-risk survival baselines
|   |-- scripts/               # Python CLIs for benchmarks and statistical analysis
|   |-- utils/                 # Config, logging, I/O, Optuna, and reproducibility helpers
|-- tests/                     # Unit and integration tests
```

## Core Model Families

SurvFM includes:

- Classical survival baselines: Cox proportional hazards and Kaplan-Meier.
- Tree-based baselines: Random Survival Forests and Gradient Boosting Survival Analysis.
- Deep survival baselines: DeepSurv, DeepHit, MTLR, PC-Hazard, SurvTRACE, SODEN, and DySurv.
- Tabular foundation model adapters: TabPFN, TabDPT, and TabICL with survival heads.
- Zero-shot in-context survival methods using single-context or per-bin time discretisation.
- Competing-risk variants for classical, deep, and foundation-model approaches.

Available registry names can be inspected with:

```python
from survpfn.models import ALL_MODELS

print(sorted(ALL_MODELS))
```

## Citation

If you use SurvFM, please cite:

```bibtex
@article{pham2026tabular,
  title={Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation},
  author={Pham, Minh-Khoi and Cotugno, Luca and Sirbu, Alina and Mai, Tai Tan and Crane, Martin and Bezbradica, Marija},
  journal={arXiv preprint arXiv:2606.12006},
  year={2026}
}
```

This paper was accepted to Artificial Intelligence in Healthcare (AIiH) 2026, London, UK.
