# Code Review — survpfn

**Reviewer:** Claude (claude-sonnet-4-6)
**Date reviewed:** 2026-03-23
**Scope:** `Dataset_analysis.ipynb`, `dataset_analysis.py`, `main.py`,
`survpfn/` package, and the new `src/` module created in this review.

---

## Integration Notes

### Changes merged from `src/` into `survpfn/` (2026-03-23)

**`survpfn/metrics.py`**
- Added `d_calibration()` function (Haider et al., 2020 algorithm) — returns mean
  squared deviation from uniform density for uncensored subjects; lower = better.
- Added `_safe_time_grid()` helper for IBS integration over the full shared support
  (replaces the narrower `valid_times`-only integration that underestimated IBS).
- `evaluate_survival_model()` now also reports `"D-cal"` in the returned dict and
  uses the full time grid for IBS. Signature is backward-compatible.
- `run_statistical_tests()` now also returns `"Median_Diff"` column.

**`survpfn/preprocessing.py`**
- `clean_and_impute()` now uses **column-stratified imputation** (median for
  continuous columns, mode for binary/ordinal) instead of `dropna()`, recovering
  ~22% of rows previously discarded.
- Hard-coded row patches (8143, 7285) are guarded with index-existence checks to
  avoid KeyError on already-reset DataFrames.
- `prepare_*` functions now use the shared `LEAKY_DATE_COLS`, `LEAKY_EVENT_FLAGS`,
  `MORTALITY_LEAKAGE_COLS` constants with `errors="ignore"` drops, eliminating
  silent failures when a column was already absent.
- `_scale_continuous()` and `_infer_binary_cols()` helpers added for optional
  in-function scaling (not used by main.py, which handles scaling externally).

**`survpfn/deep_surv.py` — Bug fix**
- `train_deepsurv()` was returning `(model, surv_df, ev)` which did not match the
  unified API, causing `ev.concordance_td()` to be reported as the C-index metric
  (returning what pycox calls "concordance_td" — actually the Antolini C-index, but
  the return type mismatch meant main.py was assigning a float to the wrong key and
  no IBS/AUC metrics were computed).
- Fixed to return `(model, risk_scores, surv_probs, surv_times)` matching all other
  model training functions. `main.py` updated accordingly.

**`survpfn/embedding_cox.py` — Bug fixes**
- `predict_survival()` had unreachable dead-code duplicate return (lines 149–150).
  Removed.
- `train_embedding_cox()` was passing the raw event column (which has values 0/1/2
  for competing-risk tasks) as the `y` label to `get_tabpfn_embeddings()`. TabPFN
  classifier requires binary labels; for competing-risk tasks this caused a
  3-class fit that produced degenerate embeddings. Fixed to binarize with
  `(event > 0).astype(int)` before calling `get_tabpfn_embeddings()`.

**`main.py` — Bug fix**
- Line 152: `plt.savefig("results/model_comparison_cindex.pdf")` was called with
  no active figure (the C-index figure was never created — only the AUROC figure
  was created below it). Fixed by wrapping the C-index plot in its own
  `fig, ax = plt.subplots()` block with proper `fig.savefig()` / `plt.close(fig)`.

### New files added

**`survpfn/benchmark_datasets.py`**
- Loaders for SUPPORT2, METABRIC, GBSG via `pycox.datasets`.
- Each `load_<dataset>()` returns `(X, y_time, y_event)` as numpy arrays with
  StandardScaler + one-hot encoding applied.
- `BENCHMARK_DATASETS` dict provides a registry for the benchmark runner.
- Requires: `uv add pycox`

**`survpfn/discrete_surv.py`**
- `train_mtlr()` — Multi-Task Logistic Regression for Survival (pycox MTLR).
- `train_pchazard()` — Piecewise-constant hazard / logistic-hazard (pycox PCHazard).
- `train_deephit_single()` — Discrete-time single-risk model (pycox DeepHitSingle).
- All three follow the unified `(model, risk_scores, surv_probs, surv_times)` API.
- All three support `tune=True` Optuna hyperparameter search.
- Requires: `uv add pycox torchtuples optuna`

**`benchmark.py`**
- CLI entry point for external benchmark evaluation.
- Usage: `uv run python benchmark.py --datasets SUPPORT2 METABRIC GBSG --models all`
- Runs 5-fold stratified CV; saves per-dataset CSV to `results/benchmark_<name>.csv`.
- Supports `--tune`, `--trials`, `--folds`, `--seed`, `--output-dir` flags.
- Individual model imports are lazy to avoid hard dependency failures.

---

---

## 1. Bugs Found

### Bug 1 — Data Leakage: outcome-adjacent date columns used as covariates
**Severity:** Critical
**Location:** `dataset_analysis.py` lines ~114-123; `survpfn/preprocessing.py`
`prepare_cox_data()`, `prepare_cardiovascular_data()`

The columns `CABG `, `PCI`, `Non Fatal AMI (Follow-Up)`, and `Ictus` are
converted from raw dates to *days-from-baseline* and then **passed directly
into the feature matrix**. These columns encode the date of future events that
happen after the blood-draw baseline. Including them as covariates lets the
model "see" whether a patient later received a CABG procedure, which perfectly
predicts their cardiac outcome.

Similarly, `Data of death` (days to death) is kept in `cols_to_keep` in
`prepare_cardiovascular_data()` and only dropped later — but the scaler is
fitted on the frame that still includes it, meaning the scaler's mean/variance
absorbs death-time information.

**Fix applied in `src/data.py`:**
- `LEAKY_DATE_COLS` and `LEAKY_EVENT_FLAGS` constants enumerate every
  outcome-adjacent column.
- All `prepare_*` functions drop these columns *before* calling `_scale_continuous`.
- `MORTALITY_LEAKAGE_COLS` additionally lists cause-of-death sub-categories
  (`CVD Death`, `Fatal MI or Sudden death`, etc.) that directly decompose the
  target label.

---

### Bug 2 — Data Leakage: Scaler fitted on the full dataset before train/test split
**Severity:** Critical
**Location:** `dataset_analysis.py` lines 183-187, 419-423, 629-631, 736-738;
`survpfn/preprocessing.py` in all `prepare_*` helpers

In the notebook and in the extracted script, `StandardScaler.fit_transform` is
called on the entire dataset (`df_mortality`, `df_cardiovascular`, etc.)
**before** the train/test split on lines 329-331. The scaler therefore sees the
distribution of the test set, which inflates out-of-sample performance metrics
and makes proper cross-validation results unreliable.

`main.py` has a correct inner-fold scaler (lines 83-84) but this is negated
because the `prepare_*` functions each apply a second `fit_transform` on the
full fold-level data inside themselves.

**Fix applied in `src/data.py` + `src/train.py`:**
- Every `prepare_*` function accepts an optional `scaler` argument. When
  `scaler=None` (training fold), it fits a new `StandardScaler` and returns it.
  When a pre-fitted scaler is passed (test fold), it calls `.transform()` only.
- `cross_validate()` in `src/train.py` calls `prepare_func(df_train, scaler=None)`
  to get the fold-level scaler, then passes it to `prepare_func(df_test, scaler=fold_scaler)`.

---

### Bug 3 — Aggressive `dropna` discards ~22% of data
**Severity:** High
**Location:** `dataset_analysis.py` line 140; `survpfn/preprocessing.py` line 43

```python
df_main = df_main.dropna()   # drops all rows with any missing value
```

The notebook prints the missingness table showing several columns with 10–20%
missing rates. Dropping all rows with any NaN removes approximately 22% of the
cohort, which:
1. Reduces statistical power.
2. Introduces systematic bias if missingness is not completely at random (MCAR).
3. May preferentially drop patients with the most severe outcomes.

**Fix applied in `src/data.py` `clean_and_impute()`:**
- Binary / ordinal columns use mode imputation.
- Continuous columns use median imputation.
- Rows with residual non-numeric NaNs are dropped *after* targeted imputation,
  with a `warnings.warn` so the caller knows.
- The only unconditional `dropna` is on the `Number` column (patient identifier)
  to remove the small set of wholly invalid rows.

---

### Bug 4 — No cross-validation; single 80/20 split
**Severity:** High
**Location:** `dataset_analysis.py` lines 329-331; all model evaluation blocks
in notebook; `survpfn/deep_surv.py`, `survpfn/deep_hit.py`

All models are trained and evaluated on a single random 80/20 split. With
typical clinical cohort sizes of 1 000–3 000 patients, a single split gives
highly variable C-index estimates (95% CI ± 0.04–0.08 is common). There is no
way to distinguish a truly better model from a lucky test fold.

**Fix applied in `src/train.py`:**
- `cross_validate()` implements stratified k-fold CV (stratified on binary
  event indicator to equalise event rates across folds).
- `make_kfold_splits()` in `src/data.py` returns fold index pairs for use
  outside `cross_validate`.

---

### Bug 5 — Only C-index reported; no calibration or IBS
**Severity:** Medium
**Location:** All model evaluation cells in `Dataset_analysis.ipynb`;
`main.py` lines 101, 106, 112

The C-index measures discrimination (ranking) but not calibration (whether
predicted probabilities match observed frequencies). A model can have a C-index
of 0.75 while being wildly miscalibrated, which is a serious problem for
clinical use.

**Fix applied in `src/metrics.py`:**
- **Integrated Brier Score (IBS)**: proper scoring rule; lower is better.
  Uses the full dense time grid rather than sparse percentile points.
- **Time-dependent AUROC** (cumulative dynamic AUC via `sksurv`): AUC at
  specific time horizons plus a mean over the evaluation window.
- **D-calibration** (Haider et al., 2020): checks that the predicted survival
  probability at each event time follows Uniform(0,1) for uncensored subjects.
- **Cause-specific C-index** (`competing_risk_cindex`): per-cause concordance
  for competing-risk tasks.

---

### Bug 6 — `DeepHitSingle` used for multi-cause competing risks
**Severity:** Medium
**Location:** `dataset_analysis.py` lines 490, 579-583; `survpfn/deep_hit.py`
line 13

`DeepHitSingle` is designed for single-event data (event ∈ {0, 1}). The
cardiovascular and MI tasks use event ∈ {0, 1, 2}. Passing multi-class event
codes directly to `EvalSurv.concordance_td()` treats event type 2 as an
ordinary censored observation, silently producing misleading C-index values.

The correct approach is either:
(a) `pycox.models.DeepHit` (full competing-risk model), or
(b) cause-specific binary C-index evaluation.

**Partially mitigated in `src/models.py`:**
`DeepHitWrapper.predict_risk()` returns the CIF at the last time point (a
cause-agnostic ranking). `src/metrics.py` `competing_risk_cindex()` evaluates
per-cause C-index by binarising event codes.

**Remaining TODO:** Replace with `pycox.models.DeepHit` for proper competing-risk
survival estimation.

---

### Bug 7 — MI endpoint: index misalignment between `df_MI` and `df_main`
**Severity:** Medium
**Location:** `dataset_analysis.py` lines 635-649

```python
conditions = [
    df_main["Non Fatal AMI (Follow-Up)_event"] == 1,   # df_main index
    df_main["Fatal MI or Sudden death"] == 1
]
choices = [
    df_main["Non Fatal AMI (Follow-Up)"],
    df_main["Data of death"]
]
df_MI["MI_date"] = np.select(conditions, choices,      # df_MI index
                              default=df_main["Follow Up Data"])
```

`df_MI` may have a different index from `df_main` after rows were dropped in
`clean_and_impute`. `np.select` operates positionally, not by index label, so
event dates can be silently assigned to the wrong patients for any patient
after a dropped row.

**Fix applied in `src/data.py` `prepare_mi_data()`:** All operations use
`df.copy()` from the already-cleaned and index-reset frame; `np.select`
operates on the same DataFrame throughout.

---

### Bug 8 — Duplicate unreachable code in `EmbeddingCoxPH.predict_survival`
**Severity:** Low
**Location:** `survpfn/embedding_cox.py` lines 148-150

```python
    def predict_survival(self, embeddings):
        ...
        return self.model.predict_surv_df(x)   # line 147

        x = embeddings.astype(np.float32)      # line 149 — unreachable
        return self.model.predict_surv_df(x)   # line 150 — unreachable
```

Two lines after the `return` are dead code (copy-paste error). No functional
impact.

---

### Bug 9 — `_init_weights` in `MLPVanilla` silently no-ops
**Severity:** Low
**Location:** `survpfn/embedding_cox.py` lines 52-58

```python
for i, m in enumerate(self.net):
    if isinstance(m, nn.Linear) and m is not list(self.net.children())[-1]:
```

The guard `m is not list(self.net.children())[-1]` rebuilds the list on every
iteration and uses identity (`is`) comparison. A new `list()` call always
returns a new object, so `m is not new_list[-1]` is almost always `True`,
meaning the output layer's weights are always re-initialised despite the
developer's intent to skip it. The commented-out simpler form below was correct.

---

### Bug 10 — `main.py:152` calls `plt.savefig` with no open figure
**Severity:** Low
**Location:** `main.py` line 152

```python
plt.savefig("results/model_comparison_cindex.pdf")
```

This line is called after `run_statistical_tests()` without any preceding
`plt.figure()` or `sns.barplot()`. It saves a blank PDF (or the last figure
if one happens to be open from a previous iteration).

**Fix applied in `src/evaluate.py`:** All plotting is encapsulated in
`_plot_metric_comparison()` and `_plot_heatmap()` which each create, save, and
close their own figures.

---

### Bug 11 — `tabpfn_modeling/utils.py` uses `torch.load` without `weights_only`
**Severity:** Low (security / deprecation)
**Location:** `survpfn/tabpfn_modeling/utils.py`

`torch.load(path, map_location='cpu')` without `weights_only=True` allows
arbitrary code execution when loading from untrusted model files. This triggers
a deprecation warning in PyTorch ≥ 2.0 and will be an error in a future release.

---

### Bug 12 — `aware_cox.py` TabPFN forward pass omits causal masking
**Severity:** Critical (for `aware_cox.py` only)
**Location:** `survpfn/aware_cox.py` lines 120-145

`TabPFNCoxModel.forward()` manually replicates the TabPFN transformer forward
pass but sets `style_src = torch.tensor([])` and never calls
`generate_D_q_matrix` for causal masking. This is required for in-context
learning correctness in TabPFN. The model may appear to run but will produce
incorrect embeddings.

Note: `aware_cox.py` is not called from `main.py`, so this does not affect
current benchmark results.

---

## 2. Improvements Made

### 2.1 `pyproject.toml`

**Before:**
```toml
requires-python = "==3.12.12"   # exact pin, prevents any other Python 3.12.x
```

**After (updated in-place):**
- `requires-python = ">=3.10"` — lower bound instead of exact pin.
- All dependency version pins changed from exact to lower-bound (`>=`).
- Added missing packages: `matplotlib`, `scipy` (Wilcoxon test), `jupyter`,
  `ipykernel`.
- Added `[project.optional-dependencies]` `dev` group (pytest, ruff, mypy).
- Added `[project.scripts]` entry point `survpfn-eval` mapping to `src.evaluate:main`.
- Added `[build-system]` (was missing) and `[tool.hatch.build]` to include
  both `survpfn/` and `src/`.
- Added `[tool.ruff]` for consistent code style enforcement.

### 2.2 `src/data.py` — Clean, typed data module

Replaces the notebook cleaning section and the partial `survpfn/preprocessing.py`.

Key additions:
- All public functions are fully type-annotated with Google-style docstrings.
- `clean_and_impute()` replaces the global `dropna()` with column-stratified
  imputation (see Bug 3).
- Every `prepare_*` returns `(df, scaler)` to support fold-level scaling.
- `make_splits()` and `make_kfold_splits()` provide reusable split utilities.
- `LEAKY_DATE_COLS`, `LEAKY_EVENT_FLAGS`, `MORTALITY_LEAKAGE_COLS` constants
  make the leakage-prevention policy explicit and reviewable.
- `_scale_continuous()` helper separates binary columns from continuous ones
  before scaling, matching the notebook's intent.

### 2.3 `src/metrics.py` — Extended metric suite

- `evaluate_survival_model()` computes C-index, IBS, time-dependent AUROC,
  and D-calibration in a single call using a shared dense time grid.
- `_safe_time_grid()` prevents the common failure mode where evaluation times
  fall outside the training support.
- `d_calibration()` is a standalone function implementing the Haider et al.
  (2020) algorithm.
- `competing_risk_cindex()` handles multi-cause event codes correctly.
- `summarise_results()` aggregates fold-level DataFrame to mean ± std tables
  ready for publication.
- All failures are caught with `warnings.warn` rather than silently passing
  or crashing.

### 2.4 `src/models.py` — Unified model API

Five wrappers sharing `fit(df, duration_col, event_col)`,
`predict_risk(df)`, `predict_survival(df)`:

| Class | Backend |
|-------|---------|
| `CoxPHWrapper` | lifelines `CoxPHFitter` |
| `DeepSurvWrapper` | pycox `CoxPH` (DeepSurv) |
| `DeepHitWrapper` | pycox `DeepHitSingle` |
| `RSFWrapper` | sksurv `RandomSurvivalForest` |
| `GBSAWrapper` | sksurv `GradientBoostingSurvivalAnalysis` |

All wrappers accept the same `(df, duration_col, event_col)` arguments so
they can be swapped in the CV loop without code changes.

### 2.5 `src/train.py` — Leakage-free k-fold training loop

- `train_one_fold()` trains and evaluates a single model on one split.
- `cross_validate()` loops over folds, fitting the scaler **inside** each fold.
- `run_all_tasks()` is a convenience wrapper for the multi-task benchmark.
- Model factories (zero-argument callables) rather than shared instances ensure
  no state persists between folds for stateful models.

### 2.6 `src/evaluate.py` — End-to-end pipeline with CLI

- `run_evaluation_pipeline()` is the single entry point for the full benchmark.
- `default_model_factories()` returns the full set of supported models.
- `DEFAULT_TASKS` mirrors the four analysis tasks from the notebook.
- Bar charts (per-metric, per-task) and heatmaps of C-index / IBS / AUC
  saved as PDFs.
- `main()` provides a `python -m src.evaluate` CLI with `--folds`, `--models`,
  `--tasks`, `--seed` arguments.

---

## 3. Remaining TODOs

### High Priority

- [ ] **Replace `DeepHitSingle` with `pycox.models.DeepHit`** for CV Mortality,
  MI, and Stroke tasks. `DeepHitSingle` is for single events only.

- [ ] **Validate imputation quality**: Run a sensitivity analysis comparing
  complete-case analysis vs. imputed dataset (e.g. use
  `sklearn.impute.IterativeImputer` as a stronger alternative to median/mode).

- [ ] **Fix `TabPFNCoxModel.forward()` masking** in `survpfn/aware_cox.py`
  (Bug 12) before using this model.

- [ ] **Add global random seeds** (`torch.manual_seed`, `np.random.seed`) at
  the start of the training loop to ensure fold-level reproducibility.

### Medium Priority

- [ ] **URRAH dataset integration**: `dataset_analysis.py` loads
  `URRAH_TG_conLegenda.xlsx` but no preprocessing or modelling pipeline exists
  for it. Add `load_urrah_data()` to `src/data.py`.

- [ ] **MIMIC-III pipeline**: `data_mimicpipe.py` imports from an external
  `ehrmeds` package not in `pyproject.toml`. Either add the dependency or
  remove the file.

- [ ] **Optuna hyperparameter tuning** hooks in `src/models.py` wrappers.
  Currently only available in `survpfn/deep_surv.py` etc. but not wired into
  the new unified API.

- [ ] **Early stopping for neural networks**: `DeepSurvWrapper` and
  `DeepHitWrapper` train for a fixed epoch count. Add validation-based early
  stopping using pycox callbacks.

- [ ] **Nested cross-validation**: For proper HP tuning evaluation, the Optuna
  search should happen inside the training fold (inner loop), not on a fixed
  held-out split.

- [ ] **Bootstrap confidence intervals** around fold-mean metrics.

### Low Priority

- [ ] Fix unreachable dead code in `survpfn/embedding_cox.py:149-150` (Bug 8).
- [ ] Fix `_init_weights` identity comparison bug in `embedding_cox.py` (Bug 9).
- [ ] Add `weights_only=True` to `torch.load` in `tabpfn_modeling/utils.py` (Bug 11).
- [ ] Add `__init__.py` to `survpfn/` (currently relies on implicit namespace packages).
- [ ] Add unit tests for `src/data.py` (imputation, leakage exclusion) and
  `src/metrics.py` (IBS, D-cal on synthetic data).
- [ ] Switch `tabpfn/__init__.py` to `TabPFNEmbedding(n_fold=5)` for
  out-of-fold train embeddings (improves generalisation per the cited paper).

---

## 4. Architecture of the New `src/` Module

```
survpfn/
├── pyproject.toml           # updated: bounds, build config, ruff, entry point
├── src/
│   ├── __init__.py          # package docstring
│   ├── data.py              # loading, cleaning, imputation, task splits
│   ├── metrics.py           # C-index, IBS, AUC, D-cal, Wilcoxon tests
│   ├── models.py            # CoxPH, DeepSurv, DeepHit, RSF, GBSA wrappers
│   ├── train.py             # k-fold CV loop, multi-task runner
│   └── evaluate.py          # end-to-end pipeline + CLI entry point
└── survpfn/                 # original package (kept for TabPFN embedding work)
    ├── data_loader.py
    ├── preprocessing.py     # leakage issues documented above
    ├── metrics.py
    ├── cox_models.py
    ├── deep_surv.py
    ├── deep_hit.py
    ├── tree_models.py
    ├── custom_model.py
    ├── embedding_cox.py     # duplicate return (Bug 8), _init_weights bug (Bug 9)
    ├── aware_cox.py         # TabPFN forward masking bug (Bug 12)
    ├── plotting.py
    └── tabpfn/
```

### Data flow

```
load_and_merge_data()
        │
        ▼
clean_and_impute()          ← median/mode imputation; no global dropna
        │
        ├─── prepare_cox_data(scaler=None)      ┐
        ├─── prepare_cardiovascular_data(...)   ├── called inside CV fold
        ├─── prepare_mi_data(...)               │   with fold-level scaler
        └─── prepare_stroke_data(...)           ┘
                │
                ▼
        cross_validate()
          for fold in folds:
            df_train ──► prepare_func(scaler=None) → df_train_scaled + scaler
            df_test  ──► prepare_func(scaler=fold_scaler) → df_test_scaled
            for model in models:
              model.fit(df_train_scaled, ...)
              risk = model.predict_risk(df_test_scaled)
              surv_probs, surv_times = model.predict_survival(df_test_scaled)
              metrics = evaluate_survival_model(...)
                │
                ▼
        pd.DataFrame(fold_results)
                │
                ├─── summarise_results()     → mean ± std CSV
                ├─── run_statistical_tests() → Wilcoxon p-values CSV
                └─── _plot_*()              → PDFs
```

### Key design invariants

| Invariant | Where enforced |
|-----------|---------------|
| Scaler fitted only on training fold | `cross_validate()` in `src/train.py` |
| Leaky columns excluded before scaling | `prepare_*()` in `src/data.py` |
| Event-stratified folds | `make_kfold_splits()` (StratifiedKFold on binary event) |
| Models stateless across folds | Factory callables in `run_all_tasks()` |
| Time grid within train/test support | `_safe_time_grid()` in `src/metrics.py` |
| All metric failures caught | try/except + `warnings.warn` in metrics functions |

---

*End of review.*
