"""
scripts/benchmark.py — Unified survival benchmark runner for the SurvPFN project.

Runs 5-fold CV across all supported datasets and models. For each
(dataset, model, fold), creates an independent result folder:

    results/<DATASET>/<model>/fold_<N>/
        metrics.json            — C-index, IBS, AUC, D-cal
        feature_importance.json — model-specific feature weights (if available)
        best_params.json        — best Optuna hyperparameters + tuning summary
        optuna_<model>.log      — Optuna journal (tunable models only)
        metadata.json           — run config, timing, dataset split stats

No aggregation is performed here. Use scripts/aggregate.py to collect all
result folders into a single CSV, then xai/plot_comparison.py for figures.

CLI
---
    # All public datasets, all models
    uv run python -m survpfn.scripts.benchmark \\
        --datasets SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN

    # Selected models with tuning
    uv run python -m survpfn.scripts.benchmark \\
        --datasets GBSG --models cox rsf gbsa --tune --trials 20

    # TabPFN jointly-trained models
    uv run python -m survpfn.scripts.benchmark \
        --datasets GBSG --models tabpfn_cox tabpfn_deephit \
        --epochs 100 --lr 1e-3 --device cuda:0

Available model names
---------------------
  cox, km
  rsf, gbsa
  deepsurv, mtlr, pchazard, deephit_single
  tabpfn_embedding_cox, tabpfn_embedding_deephit, tabpfn_embedding_pchazard, tabpfn_embedding_mtlr
                                               (TabPFN frozen → survival head)
  tabpfn_cox, tabpfn_deephit, tabpfn_pchazard, tabpfn_mtlr
                                               (TabPFN jointly trained)
  tabdpt_embedding_{cox,deephit,pchazard,mtlr} (TabDPT frozen; needs TABDPT_CHECKPOINT)
  tabicl_embedding_{cox,deephit,pchazard,mtlr} (TabICL frozen; auto-downloads checkpoint)
  all  (runs all of the above)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from survpfn.dataloaders import BENCHMARK_DATASETS
from survpfn.metrics.metrics import evaluate_survival_model


# ---------------------------------------------------------------------------
# Fold-level data preparation
# ---------------------------------------------------------------------------

def _scale_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    duration_col: str,
    event_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split, scale features (fit on train only), binarize competing risks.

    Competing-risk event columns (values > 1) are binarized to {0, 1} so that
    single-risk models can be applied uniformly across all datasets.
    """
    feat_cols = [c for c in df.columns if c not in {duration_col, event_col}]

    df_train = df.iloc[train_idx].reset_index(drop=True).copy()
    df_test = df.iloc[test_idx].reset_index(drop=True).copy()

    # Binarize competing-risk events
    if df[event_col].max() > 1:
        df_train[event_col] = (df_train[event_col] > 0).astype(float)
        df_test[event_col] = (df_test[event_col] > 0).astype(float)

    scaler = StandardScaler()
    df_train[feat_cols] = scaler.fit_transform(df_train[feat_cols])
    df_test[feat_cols] = scaler.transform(df_test[feat_cols])

    return df_train, df_test


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def get_feature_importance(
    model_name: str,
    model,
    feature_names: list[str],
) -> dict | None:
    """Return a serialisable feature-importance dict or None if not available.

    Supported models
    ----------------
    cox  — coefficients + hazard ratios + p-values from ``CoxPHFitter.summary``
    rsf  — impurity-based feature importances from ``RandomSurvivalForest``
    gbsa — impurity-based feature importances from ``GradientBoostingSurvivalAnalysis``
    """
    if model is None:
        return None

    try:
        if model_name in ("rsf", "gbsa") and hasattr(model, "feature_importances_"):
            return {
                "type": "impurity",
                "features": dict(zip(feature_names, model.feature_importances_.tolist())),
            }

        if model_name == "cox" and hasattr(model, "summary"):
            s = model.summary.reset_index()
            feat_col = "covariate" if "covariate" in s.columns else s.columns[0]
            feats = s[feat_col].tolist()
            return {
                "type": "cox_coefficients",
                "coef":     dict(zip(feats, s["coef"].tolist())),
                "exp_coef": dict(zip(feats, np.exp(s["coef"]).tolist())),
                "p":        dict(zip(feats, s["p"].tolist())),
            }
    except (NotImplementedError, Exception):
        return None

    return None


# ---------------------------------------------------------------------------
# Optuna best-params extraction
# ---------------------------------------------------------------------------

# Maps model_name → (log_filename, optuna study_name) matching each model module.
_OPTUNA_STUDY: dict[str, tuple[str, str]] = {
    "rsf":            ("optuna_rsf.log",            "rsf_tuning"),
    "gbsa":           ("optuna_gbsa.log",            "gbsa_tuning"),
    "deepsurv":       ("optuna_deepsurv.log",        "deepsurv_tuning"),
    "mtlr":           ("optuna_mtlr.log",            "mtlr_tuning"),
    "pchazard":       ("optuna_pchazard.log",        "pchazard_tuning"),
    "deephit_single": ("optuna_deephit_single.log",  "deephit_single_tuning"),
    # TabPFN frozen embedding heads
    "tabpfn_embedding_cox":          ("optuna_tabpfn_cox.log",           "tabpfn_cox"),
    "tabpfn_embedding_deephit":      ("optuna_tabpfn_deephit.log",       "tabpfn_deephit"),
    "tabpfn_embedding_pchazard":     ("optuna_tabpfn_pchazard.log",      "tabpfn_pchazard"),
    "tabpfn_embedding_mtlr":         ("optuna_tabpfn_mtlr.log",          "tabpfn_mtlr"),
    # TabDPT frozen embedding heads
    "tabdpt_embedding_cox":      ("optuna_tabdpt_cox.log",        "tabdpt_cox"),
    "tabdpt_embedding_deephit":  ("optuna_tabdpt_deephit.log",    "tabdpt_deephit"),
    "tabdpt_embedding_pchazard": ("optuna_tabdpt_pchazard.log",   "tabdpt_pchazard"),
    "tabdpt_embedding_mtlr":     ("optuna_tabdpt_mtlr.log",       "tabdpt_mtlr"),
    # TabICL frozen embedding heads
    "tabicl_embedding_cox":      ("optuna_tabicl_cox.log",        "tabicl_cox"),
    "tabicl_embedding_deephit":  ("optuna_tabicl_deephit.log",    "tabicl_deephit"),
    "tabicl_embedding_pchazard": ("optuna_tabicl_pchazard.log",   "tabicl_pchazard"),
    "tabicl_embedding_mtlr":     ("optuna_tabicl_mtlr.log",       "tabicl_mtlr"),
}


def get_best_params(model_name: str, out_dir: str) -> dict | None:
    """Read best hyperparameters + tuning summary from the Optuna journal in out_dir.

    Returns a dict with keys:
        best_params, best_value, n_completed_trials, best_trial_number
    Returns None if the model is not tunable or the journal does not exist yet.
    """
    if model_name not in _OPTUNA_STUDY:
        return None
    log_name, study_name = _OPTUNA_STUDY[model_name]
    log_file = os.path.join(out_dir, log_name)
    if not os.path.exists(log_file):
        return None
    try:
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = JournalStorage(JournalFileBackend(log_file))
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        return {
            "best_params":        study.best_params,
            "best_value":         study.best_value,
            "n_completed_trials": len(completed),
            "best_trial_number":  study.best_trial.number,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Neural-network parameter counter
# ---------------------------------------------------------------------------

def count_params(model) -> int | None:
    """Return the number of trainable parameters for PyTorch / pycox models.

    Returns None for non-neural models (Cox, RSF, GBSA, KM).
    """
    try:
        import torch.nn as nn
        # Direct nn.Module (e.g. TabPFNSurvPH)
        if isinstance(model, nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # pycox / torchtuples wrappers expose .net
        if hasattr(model, "net") and isinstance(model.net, nn.Module):
            return sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Per-model training functions
# All return (model_obj, risk_scores, surv_probs, surv_times).
# ---------------------------------------------------------------------------

def _train_cox(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_train, duration_col=dur_col, event_col=ev_col)

    X_test = df_test.drop(columns=[dur_col, ev_col])
    risk = cph.predict_partial_hazard(X_test).values

    t_events = df_train.loc[df_train[ev_col] > 0, dur_col].values
    t_grid = (np.percentile(t_events, np.linspace(5, 95, 50))
              if len(t_events) > 0
              else np.linspace(df_train[dur_col].min(), df_train[dur_col].max(), 50))
    surv_df = cph.predict_survival_function(X_test, times=t_grid)
    return cph, risk, surv_df.values.T, t_grid


def _train_km(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from sksurv.nonparametric import kaplan_meier_estimator
    t_km, s_km = kaplan_meier_estimator(
        df_train[ev_col].astype(bool), df_train[dur_col]
    )
    risk = np.zeros(len(df_test))
    surv_probs = np.tile(s_km, (len(df_test), 1))
    return None, risk, surv_probs, t_km


def _train_rsf(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from survpfn.models.tree import train_rsf
    return train_rsf(df_train, df_test, dur_col, ev_col,
                     tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None)


def _train_gbsa(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from survpfn.models.tree import train_gbsa
    return train_gbsa(df_train, df_test, dur_col, ev_col,
                      tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None)


def _train_deepsurv(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from survpfn.models.deep_surv import train_deepsurv
    return train_deepsurv(df_train, df_test, dur_col, ev_col,
                          tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None)


def _train_mtlr(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from survpfn.models.deep_hit import train_mtlr
    return train_mtlr(df_train, df_test, dur_col, ev_col,
                      tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None)


def _train_pchazard(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from survpfn.models.deep_hit import train_pchazard
    return train_pchazard(df_train, df_test, dur_col, ev_col,
                          tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None)


def _train_deephit_single(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **_):
    from survpfn.models.deep_hit import train_deephit_single
    return train_deephit_single(df_train, df_test, dur_col, ev_col,
                                tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None)


def _train_fm_embedding(
    fm: str,
    head: str,
    df_train, df_test, dur_col, ev_col,
    tune, n_trials, out_dir,
    **kw,
):
    """Generic dispatcher: FM backbone × survival head.

    fm   : "tabpfn" | "tabdpt" | "tabicl"
    head : "cox" | "deephit" | "pchazard" | "mtlr"
    """
    if fm == "tabpfn":
        from survpfn.models.tabpfn.cox import train_embedding_surv
        return train_embedding_surv(
            df_train, df_test, dur_col, ev_col,
            head_type=head,
            tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None,
        )
    if fm == "tabdpt":
        from survpfn.models.tabdpt.cox import train_tabdpt_embedding_surv
        return train_tabdpt_embedding_surv(
            df_train, df_test, dur_col, ev_col,
            head_type=head,
            tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None,
            checkpoint_path=kw.get("tabdpt_checkpoint", None),
            context_size=kw.get("tabdpt_context_size", None),
            use_retrieval=kw.get("tabdpt_use_retrieval", True),
            device=kw.get("device", None),
        )
    if fm == "tabicl":
        from survpfn.models.tabicl.cox import train_tabicl_embedding_surv
        return train_tabicl_embedding_surv(
            df_train, df_test, dur_col, ev_col,
            head_type=head,
            tune=tune, n_trials=n_trials, save_dir=out_dir, study_id=None,
            device=kw.get("device", None),
        )
    raise ValueError(f"Unknown FM backbone: {fm!r}")


def _train_tabpfn_surv(
    head_type: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    **kwargs,
):
    """Shared implementation for jointly-trained TabPFN survival heads."""
    from survpfn.models.tabpfn.cox import TabPFNSurvPH
    model = TabPFNSurvPH(
        head_type=head_type,
        learning_rate=kwargs.get("lr", 1e-3),
        alpha=kwargs.get("alpha", 1.0),
        device=kwargs.get("device", "cuda:0"),
    )
    model.fit(
        x=df_train.drop(columns=[dur_col, ev_col]).values,
        durations=df_train[dur_col].values,
        events=df_train[ev_col].values,
        y_pfn=(df_train[ev_col] > 0).astype(int).values,
        epochs=kwargs.get("epochs", 50),
        batch_size=kwargs.get("batch_size", 64),
        verbose=False,
    )
    surv_df = model.predict_survival_df(df_test.drop(columns=[dur_col, ev_col]).values)
    
    # Use integrated survival (AUC) as a more robust risk score.
    # Lower AUC (shorter life expectancy) = Higher risk.
    times = surv_df.index.values
    surv_array = surv_df.values  # (n_times, n_subjects)
    # Average survival over time for each subject
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise AttributeError("module 'numpy' has no attribute 'trapezoid' or 'trapz'")
    auc = _trapz(surv_array.T, times)
    risk = -auc
    return model, risk, surv_df.values.T, surv_df.index.values


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _make_fm_embedding(fm: str, head: str) -> Callable:
    """Build a benchmark-compatible callable for (fm, head) pair."""
    def _fn(df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir, **kw):
        return _train_fm_embedding(
            fm, head, df_train, df_test, dur_col, ev_col,
            tune, n_trials, out_dir, **kw,
        )
    _fn.__name__ = f"_train_{fm}_embedding_{head}"
    return _fn


ALL_MODELS: dict[str, Callable] = {
    # ── Classical ──────────────────────────────────────────────────────────
    "cox":            _train_cox,
    "km":             _train_km,
    # ── Tree ───────────────────────────────────────────────────────────────
    "rsf":            _train_rsf,
    "gbsa":           _train_gbsa,
    # ── Deep survival (classical DL heads on raw features) ─────────────────
    "deepsurv":       _train_deepsurv,
    "mtlr":           _train_mtlr,
    "pchazard":       _train_pchazard,
    "deephit_single": _train_deephit_single,
    # ── TabPFN frozen embedding × 4 heads ──────────────────────────────────
    "tabpfn_embedding_cox":         _make_fm_embedding("tabpfn", "cox"),
    "tabpfn_embedding_deephit":     _make_fm_embedding("tabpfn", "deephit"),
    "tabpfn_embedding_pchazard":    _make_fm_embedding("tabpfn", "pchazard"),
    "tabpfn_embedding_mtlr":        _make_fm_embedding("tabpfn", "mtlr"),
    # ── TabPFN jointly-trained (tabpfn_*) ────────────────────────────────────
    # positional: (df_train, df_test, dur_col, ev_col, tune, n_trials, out_dir)
    # _train_tabpfn_surv only uses the first 4; the rest forwarded via **kw
    "tabpfn_cox":       lambda *a, **kw: _train_tabpfn_surv("cox",      *a[:4], **kw),
    "tabpfn_deephit":   lambda *a, **kw: _train_tabpfn_surv("deephit",  *a[:4], **kw),
    "tabpfn_pchazard":  lambda *a, **kw: _train_tabpfn_surv("pchazard", *a[:4], **kw),
    "tabpfn_mtlr":      lambda *a, **kw: _train_tabpfn_surv("mtlr",     *a[:4], **kw),
    # ── TabDPT frozen embedding × 4 heads ──────────────────────────────────
    "tabdpt_embedding_cox":      _make_fm_embedding("tabdpt", "cox"),
    "tabdpt_embedding_deephit":  _make_fm_embedding("tabdpt", "deephit"),
    "tabdpt_embedding_pchazard": _make_fm_embedding("tabdpt", "pchazard"),
    "tabdpt_embedding_mtlr":     _make_fm_embedding("tabdpt", "mtlr"),
    # ── TabICL frozen embedding × 4 heads ──────────────────────────────────
    "tabicl_embedding_cox":      _make_fm_embedding("tabicl", "cox"),
    "tabicl_embedding_deephit":  _make_fm_embedding("tabicl", "deephit"),
    "tabicl_embedding_pchazard": _make_fm_embedding("tabicl", "pchazard"),
    "tabicl_embedding_mtlr":     _make_fm_embedding("tabicl", "mtlr"),
}


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def run_fold_model(
    dataset_name: str,
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    fold: int,
    tune: bool,
    n_trials: int,
    output_dir: str,
    **aware_kwargs,
) -> dict:
    """Train one model for one fold, save all outputs to a dedicated folder.

    Outputs
    -------
    ``<output_dir>/<dataset_name>/<model_name>/fold_<fold>/``
        metrics.json
        feature_importance.json  (when available)
        optuna.log               (when tune=True and model supports it)
        metadata.json
    """
    out_dir = os.path.join(output_dir, dataset_name, model_name, f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    feature_names = [c for c in df_train.columns
                     if c not in {duration_col, event_col}]

    # ── Train  (includes HPO when tune=True, final fit, and test-set predict) ──
    train_fn = ALL_MODELS[model_name]
    t_fit = time.perf_counter()
    model_obj, risk, surv_p, surv_t = train_fn(
        df_train, df_test, duration_col, event_col,
        tune, n_trials, out_dir, **aware_kwargs,
    )
    fit_time_s = time.perf_counter() - t_fit

    # ── Evaluate metrics ────────────────────────────────────────────────────
    t_eval = time.perf_counter()
    metrics = evaluate_survival_model(
        df_train, df_test, duration_col, event_col,
        risk, surv_probs=surv_p, surv_times=surv_t,
    )
    eval_time_s = time.perf_counter() - t_eval

    # ── Dataset split statistics ────────────────────────────────────────────
    n_events_train = int(df_train[event_col].astype(bool).sum())
    n_events_test  = int(df_test[event_col].astype(bool).sum())

    # ── Save metrics.json ───────────────────────────────────────────────────
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                   for k, v in metrics.items()}, f, indent=2)

    # ── Save feature_importance.json ────────────────────────────────────────
    fi = get_feature_importance(model_name, model_obj, feature_names)
    if fi is not None:
        with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
            json.dump(fi, f, indent=2)

    # ── Save best_params.json (read from Optuna journal written by model) ───
    if tune:
        bp = get_best_params(model_name, out_dir)
        if bp is not None:
            with open(os.path.join(out_dir, "best_params.json"), "w") as f:
                json.dump(bp, f, indent=2)

    # ── Save metadata.json ──────────────────────────────────────────────────
    meta = {
        "dataset":   dataset_name,
        "model":     model_name,
        "fold":      fold,
        "tune":      tune,
        "n_trials":  n_trials,
        # Split sizes
        "n_train":         len(df_train),
        "n_test":          len(df_test),
        "n_features":      len(feature_names),
        "n_events_train":  n_events_train,
        "n_events_test":   n_events_test,
        "event_rate_train": round(n_events_train / len(df_train), 4),
        "event_rate_test":  round(n_events_test  / len(df_test),  4),
        # Timing
        # fit_time_s: wall-clock for HPO (if tune=True) + final fit + test-set inference
        # eval_time_s: wall-clock for C-index / IBS / AUC metric computation
        "fit_time_s":              round(fit_time_s,  4),
        "eval_time_s":             round(eval_time_s, 4),
        "hpo_included_in_fit_time": tune,
        # Model complexity
        "n_params": count_params(model_obj),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    c = metrics.get("C-index", float("nan"))
    t_total = fit_time_s + eval_time_s
    print(f"      → C-index={c:.4f}  fit={fit_time_s:.1f}s"
          if not np.isnan(c)
          else f"      → FAILED  ({t_total:.1f}s)")
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_name: str,
    model_names: list[str],
    n_folds: int = 5,
    tune: bool = False,
    n_trials: int = 10,
    output_dir: str = "results",
    random_state: int = 42,
    aware_kwargs: dict | None = None,
) -> None:
    """Run k-fold CV for one dataset, saving per-fold per-model results.

    Parameters
    ----------
    dataset_name:
        Key from ``BENCHMARK_DATASETS``.
    model_names:
        List of model keys from ``ALL_MODELS``.
    n_folds:
        Number of cross-validation folds.
    tune:
        Enable Optuna hyperparameter search for tunable models.
    n_trials:
        Optuna trials per model per fold.
    output_dir:
        Root directory for per-run result folders.
    random_state:
        Seed for StratifiedKFold.
    aware_kwargs:
        Extra args forwarded to jointly-trained TabPFN models (``tabpfn_*``):
        ``epochs``, ``batch_size``, ``lr``, ``alpha``, ``device``.
    """
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    loader = BENCHMARK_DATASETS[dataset_name]
    df, dur_col, ev_col = loader()
    print(f"  Loaded: {df.shape[0]} samples | {df.shape[1] - 2} features "
          f"| event rate={df[ev_col].gt(0).mean():.3f}")

    strat_label = (df[ev_col] > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    extra = aware_kwargs or {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, strat_label)):
        fold = fold_idx + 1
        print(f"\n  --- Fold {fold}/{n_folds} ---")
        df_train, df_test = _scale_fold(df, train_idx, test_idx, dur_col, ev_col)

        for model_name in model_names:
            print(f"    [{model_name}]", end=" ", flush=True)
            # try:
            run_fold_model(
                dataset_name, model_name,
                df_train, df_test, dur_col, ev_col,
                fold, tune, n_trials, output_dir,
                **extra,
            )
            # except Exception as exc:
            #     warnings.warn(
            #         f"{dataset_name}/{model_name}/fold_{fold} failed: {exc}",
            #         stacklevel=2,
            #     )
            #     print("      → FAILED (see warning above)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SurvPFN unified benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["SUPPORT2", "METABRIC", "GBSG", "WHAS500", "VETERANS", "FLCHAIN"],
        choices=list(BENCHMARK_DATASETS.keys()),
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Models to run. 'all' or any subset of: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument("--folds",      type=int,   default=5,          help="CV folds.")
    parser.add_argument("--tune",       action="store_true",            help="Enable Optuna tuning.")
    parser.add_argument("--trials",     type=int,   default=10,         help="Optuna trials per fold.")
    parser.add_argument("--output-dir", default="results",              help="Root output directory.")
    parser.add_argument("--seed",       type=int,   default=42,         help="Random seed.")

    g = parser.add_argument_group("TabPFN jointly-trained model options (tabpfn_*)")
    g.add_argument("--epochs",     type=int,   default=50,     help="Training epochs.")
    g.add_argument("--batch-size", type=int,   default=64,     help="Mini-batch size.")
    g.add_argument("--lr",         type=float, default=1e-3,   help="Learning rate.")
    g.add_argument("--alpha",      type=float, default=1.0,    help="PFN loss weight.")
    g.add_argument("--device",     type=str,   default="cuda:0", help="Torch device.")

    args = parser.parse_args()

    if "all" in args.models:
        model_names = list(ALL_MODELS.keys())
    else:
        model_names = []
        for m in args.models:
            if m not in ALL_MODELS:
                parser.error(f"Unknown model '{m}'. Choose from: all, " + ", ".join(ALL_MODELS))
            model_names.append(m)

    aware_kwargs = {
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "alpha":      args.alpha,
        "device":     args.device,
    }

    for dataset in args.datasets:
        run_benchmark(
            dataset_name=dataset,
            model_names=model_names,
            n_folds=args.folds,
            tune=args.tune,
            n_trials=args.trials,
            output_dir=args.output_dir,
            random_state=args.seed,
            aware_kwargs=aware_kwargs,
        )


if __name__ == "__main__":
    main()
