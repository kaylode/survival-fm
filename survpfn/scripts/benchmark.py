"""
scripts/benchmark.py — Unified survival benchmark runner for the SurvPFN project.

Runs 5-fold CV across all supported datasets and models. For each
(dataset, model, fold), creates an independent result folder:

    results/<DATASET>/<model>/fold_<N>/
        metrics.json            — C_td, IBS, AUC, D-cal
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
  survtrace, deepsurv, mtlr, pchazard, deephit_single
  tabpfn_embedding_{cox,deepsurv,deephit,pchazard,mtlr}
                                               (TabPFN frozen → survival head)
  tabpfn_cox, tabpfn_deephit, tabpfn_pchazard, tabpfn_mtlr
                                               (TabPFN jointly trained)
  tabdpt_embedding_{cox,deephit,pchazard,mtlr} (TabDPT frozen; auto-downloads checkpoint)
  tabicl_embedding_{cox,deephit,pchazard,mtlr} (TabICL frozen; auto-downloads checkpoint)
  tabpfn_zeroshot, tabdpt_zeroshot, tabicl_zeroshot
                                               (zero-shot ICL; single_context mode)
  tabpfn_zeroshot_perbin, tabdpt_zeroshot_perbin, tabicl_zeroshot_perbin
                                               (zero-shot ICL; per_bin mode)
  all  (runs all of the above, excluding *_perbin variants)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Callable

# ── Color codes ──────────────────────────────────────────────────────────────
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[96m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# New imports
from survpfn.dataloaders import get_dataset
from survpfn.models import ALL_MODELS
from survpfn.metrics import evaluate_cr, evaluate_sr_survival_eval
from survpfn.utils.config import seed_everything
from survpfn.utils.logger import LoggerObserver
from survpfn.utils.optuna import (
    _scale_fold,
    get_feature_importance,
    get_best_params,
    count_params
)
from survpfn.models.shared.binning import resolve_num_durations


# ---------------------------------------------------------------------------
# Model groups for CLI selection
# ---------------------------------------------------------------------------

MODEL_GROUPS: dict[str, list[str]] = {
    "classical": ["cox", "km"],
    "tree":      ["rsf", "gbsa"],
    "deep":      ["deepsurv", "mtlr", "pchazard", "deephit_single", "survtrace", "soden"],

    # Strategy 1 — Frozen Transfer (SR)
    "fm_embedding": [
        "tabpfn_embedding_cox", "tabpfn_embedding_deephit", "tabpfn_embedding_pchazard", "tabpfn_embedding_mtlr",
        "tabdpt_embedding_cox", "tabdpt_embedding_deephit", "tabdpt_embedding_pchazard", "tabdpt_embedding_mtlr",
        "tabicl_embedding_cox", "tabicl_embedding_deephit", "tabicl_embedding_pchazard", "tabicl_embedding_mtlr",
    ],

    # Strategy 2 — Joint Adaptation (SR)
    "fm_joint": [
        "tabpfn_joint_cox", "tabpfn_joint_deephit", "tabpfn_joint_pchazard", "tabpfn_joint_mtlr",
        "tabdpt_joint_cox", "tabdpt_joint_deephit", "tabdpt_joint_pchazard", "tabdpt_joint_mtlr",
        "tabicl_joint_cox", "tabicl_joint_deephit", "tabicl_joint_pchazard", "tabicl_joint_mtlr",
    ],

    # Strategy 3 — Zero-shot ICL (SR)
    "zeroshot": [
        "tabpfn_zeroshot", "tabdpt_zeroshot", "tabicl_zeroshot",
        "tabpfn_zeroshot_perbin", "tabdpt_zeroshot_perbin", "tabicl_zeroshot_perbin",
    ],
    "zeroshot_temporal": [
        "tabpfn_zeroshot_perbin_time", "tabdpt_zeroshot_perbin_time", "tabicl_zeroshot_perbin_time",
        "tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens",
    ],

    # Strategy 4 — Temporal Finetuning
    "finetune": [
        "tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune",
    ],

    # Competing Risks
    "cr": [
        "cox_cr", "aj_cr", "fine_gray_cr", "survival_boost_cr", "deephit_cr",
        "tabpfn_embedding_deephit_cr", "tabdpt_embedding_deephit_cr", "tabicl_embedding_deephit_cr",
        "tabpfn_joint_deephit_cr", "tabdpt_joint_deephit_cr", "tabicl_joint_deephit_cr",
        "tabpfn_zeroshot_cr", "tabdpt_zeroshot_cr", "tabicl_zeroshot_cr",
    ],
}

# ── Backwards compatibility & Backbone groups ────────────────────────────────
MODEL_GROUPS["tabpfn"] = [m for m in ALL_MODELS.keys() if "tabpfn" in m]
MODEL_GROUPS["tabdpt"] = [m for m in ALL_MODELS.keys() if "tabdpt" in m]
MODEL_GROUPS["tabicl"] = [m for m in ALL_MODELS.keys() if "tabicl" in m]
MODEL_GROUPS["fm"] = MODEL_GROUPS["fm_embedding"] + MODEL_GROUPS["fm_joint"] + MODEL_GROUPS["zeroshot"] + MODEL_GROUPS["finetune"]

MODEL_GROUPS["all"] = (
    MODEL_GROUPS["classical"] +
    MODEL_GROUPS["tree"] +
    MODEL_GROUPS["deep"] +
    MODEL_GROUPS["fm"]
)


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def _save_predictions(
    pred_dir: str,
    dataset_name: str,
    model_tag: str,
    fold: int,
    test_idx: np.ndarray,
    risk_scores: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> None:
    """Save per-fold risk scores indexed by original dataset position.

    Written to: ``<pred_dir>/<dataset_name>/<model_tag>/fold_<fold>.parquet``

    Columns
    -------
    original_idx : iloc position in the original (pre-scale) DataFrame.
                   Use this to join back to raw demographics at eval time.
    risk_score   : scalar risk score — higher = higher risk of event.
    duration     : raw (unscaled) event / censoring time.
    event        : binary event indicator (0 = censored, 1 = event).
    fold         : fold index (1-based).
    """
    out = os.path.join(pred_dir, dataset_name, model_tag)
    os.makedirs(out, exist_ok=True)
    
    data = {
        "original_idx": test_idx.astype(np.int64),
        "duration":     np.asarray(duration,    dtype=np.float64),
        "event":        np.asarray(event,       dtype=np.int64),
        "fold":         fold,
    }
    
    if isinstance(risk_scores, list) or (isinstance(risk_scores, np.ndarray) and risk_scores.ndim > 1):
        for i, rs in enumerate(risk_scores):
            data[f"risk_score_c{i+1}"] = np.asarray(rs, dtype=np.float64)
    else:
        data["risk_score"] = np.asarray(risk_scores, dtype=np.float64)
        
    pd.DataFrame(data).to_parquet(os.path.join(out, f"fold_{fold}.parquet"), index=False)


def run_fold_model(
    dataset_name: str,
    model_name: str,
    model_tag: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    fold: int,
    tune: bool,
    n_trials: int,
    output_dir: str,
    test_idx: np.ndarray | None = None,
    pred_dir: str | None = None,
    tuned_dir: str | None = None,
    **training_kwargs,
) -> dict:
    """Train one model for one fold, save all outputs to a dedicated folder.

    Parameters
    ----------
    test_idx : iloc indices of test rows in the original (pre-scale) DataFrame.
               Required when ``pred_dir`` is set; used to link risk scores back
               to raw demographic columns for fairness evaluation.
    pred_dir : When set, saves ``<pred_dir>/<dataset>/<model_tag>/fold_N.parquet``
               containing original_idx, risk_score, duration, event.

    Outputs
    -------
    ``<output_dir>/<dataset_name>/<model_name>/fold_<fold>/``
        metrics.json
        feature_importance.json  (when available)
        optuna.log               (when tune=True and model supports it)
        metadata.json
    """
    out_dir = os.path.join(output_dir, dataset_name, model_tag, f"fold_{fold}")

    if training_kwargs["num_durations"] == -1:
        training_kwargs["num_durations"] = resolve_num_durations(
            df_train[duration_col].values,
            df_train[event_col].values,
            -1,
        )
    else:
        out_dir = os.path.join(out_dir, f"bins{training_kwargs['num_durations']}")

    # Determine whether a predictions file is expected (for skip-check logic)
    pred_path = None
    if pred_dir is not None and test_idx is not None:
        pred_path = os.path.join(pred_dir, dataset_name, model_tag, f"fold_{fold}.parquet")

    # ── Skip check ────────────────────────────────────────────────────────
    # Only skip when BOTH metrics are valid AND (no predictions needed OR they
    # already exist).  If predictions are missing, re-run even if metrics exist.
    metrics_file = os.path.join(out_dir, "metrics.json")
    pred_already_saved = (pred_path is None) or os.path.exists(pred_path)
    if os.path.exists(metrics_file) and pred_already_saved:
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            c = metrics.get("C_td", float("nan"))
            ibs = metrics.get("IBS", float("nan"))
            auc = metrics.get("AUC_mean", float("nan"))
            if not np.isnan(c) and not np.isnan(ibs) and not np.isnan(auc):
                print(f"      {C_BLUE}→ SKIPPING (C={c:.4f} IBS={ibs:.4f} AUC={auc:.4f}){C_RESET}")
                return metrics
            else:
                print(f"      {C_YELLOW}→ Metric is NaN, retraining{C_RESET}")
        except Exception:
            pass

    os.makedirs(out_dir, exist_ok=True)
    if tuned_dir is not None:
        src_dir = os.path.join(tuned_dir, dataset_name, model_name, f"fold_{fold}")
        from survpfn.utils.optuna import _LEGACY_STUDY_OVERRIDES
        import shutil
        import warnings
        log_name = _LEGACY_STUDY_OVERRIDES.get(model_name, (f"optuna_{model_name}.log",))[0]
        src_log = os.path.join(src_dir, log_name)
        dst_log = os.path.join(out_dir, log_name)
        if os.path.exists(src_log):
            if not os.path.exists(dst_log):
                shutil.copy(src_log, dst_log)
            tune = True
            n_trials = 0
        else:
            warnings.warn(f"Tuned logs not found at {src_log}, falling back to default behavior.")

    print()
    feature_names = [c for c in df_train.columns
                     if c not in {duration_col, event_col}]

    # ── Train  (includes HPO when tune=True, final fit, and test-set predict) ──
    train_fn = ALL_MODELS[model_name]
    t_fit = time.perf_counter()
    model_obj, risk, surv_p, surv_t = train_fn(
        df_train, df_test, duration_col, event_col,
        tune=tune, n_trials=n_trials, out_dir=out_dir, **training_kwargs,
    )
    fit_time_s = time.perf_counter() - t_fit

    # ── Save predictions for fairness evaluation ────────────────────────────
    if pred_path is not None and test_idx is not None:
        num_events_for_pred = int(df_train[event_col].max())
        if num_events_for_pred > 1 and isinstance(surv_p, list):
            risk_scores_out = []
            span = surv_t[-1] - surv_t[0] + 1e-8
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            for c_cif in surv_p:
                if _trapz is not None:
                    risk_scores_out.append(_trapz(c_cif, surv_t, axis=1) / span)
                else:
                    risk_scores_out.append(c_cif[:, -1])
        else:
            risk_scores_out = risk if not isinstance(risk, list) else risk[0]

        try:
            _save_predictions(
                pred_dir=pred_dir,
                dataset_name=dataset_name,
                model_tag=model_tag,
                fold=fold,
                test_idx=test_idx,
                risk_scores=risk_scores_out,
                duration=df_test[duration_col].values,
                event=df_test[event_col].values,
            )
        except Exception as _pred_exc:
            warnings.warn(f"Could not save predictions for {model_tag}/fold_{fold}: {_pred_exc}")

    # ── Evaluate metrics ────────────────────────────────────────────────────
    t_eval = time.perf_counter()
    num_events = int(df_train[event_col].max())
    if num_events > 1:
        # Competing risks: surv_p is list of CIFs
        metrics = evaluate_cr(
            df_train, df_test, duration_col, event_col,
            cif_per_cause=surv_p, surv_times=surv_t,
            risk_scores=risk,
        )
    else:
        # Single risk: surv_p is matrix of S(t)
        metrics = evaluate_sr_survival_eval(
            df_train, df_test, duration_col, event_col,
            surv_probs=surv_p, surv_times=surv_t,
            risk_scores=risk,
            output_dir=out_dir
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
        "model_tag": model_tag,
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
        # eval_time_s: wall-clock for C_td / IBS / AUC metric computation
        "fit_time_s":              round(fit_time_s,  4),
        "eval_time_s":             round(eval_time_s, 4),
        "hpo_included_in_fit_time": tune,
        # Model complexity
        "n_params": count_params(model_obj),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    c = metrics.get("C_td", float("nan"))
    ibs = metrics.get("IBS", float("nan"))
    auc = metrics.get("AUC_mean", float("nan"))
    t_total = fit_time_s + eval_time_s
    if not np.isnan(c):
        print(f"      {C_GREEN}→ C={c:.4f} IBS={ibs:.4f} AUC={auc:.4f}{C_RESET}  fit={fit_time_s:.1f}s")
    else:
        breakpoint()
        print(f"      {C_YELLOW}→ FAILED  ({t_total:.1f}s){C_RESET}")
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
    training_kwargs: dict | None = None,
    logger: LoggerObserver | None = None,
    use_temporal_split: bool = False,
    temporal_frac_train: float = 0.70,
    save_predictions: bool = False,
    pred_dir: str = "results/predictions",
    tuned_dir: str | None = None,
) -> None:
    """Run k-fold CV (or a single temporal split) for one dataset.

    Parameters
    ----------
    dataset_name:
        Key from ``BENCHMARK_DATASETS``.
    model_names:
        List of model keys from ``ALL_MODELS``.
    n_folds:
        Number of cross-validation folds (ignored when use_temporal_split=True).
    tune:
        Enable Optuna hyperparameter search for tunable models.
    n_trials:
        Optuna trials per model per fold.
    output_dir:
        Root directory for per-run result folders.
    random_state:
        Seed for StratifiedKFold.
    training_kwargs:
        Extra args forwarded to jointly-trained TabPFN models (``tabpfn_*``):
        ``epochs``, ``batch_size``, ``lr``, ``alpha``, ``device``.
    use_temporal_split:
        If True, replace k-fold CV with a single prospective 70/30 row-order
        split (oldest rows → train; newest rows → test).
    temporal_frac_train:
        Training fraction for the temporal split (default 0.70).
    """
    from survpfn.dataloaders import temporal_split as _temporal_split

    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    df, dur_col, ev_col = get_dataset(dataset_name)
    print(f"  Loaded: {df.shape[0]} samples | {df.shape[1] - 2} features "
          f"| event rate={df[ev_col].gt(0).mean():.3f}")

    extra = training_kwargs or {}
    label_fractions = (training_kwargs or {}).get("label_fractions", None)

    # ── Build fold iterator ──────────────────────────────────────────────────
    if use_temporal_split:
        print(f"  Split: temporal ({temporal_frac_train:.0%} train / "
              f"{1-temporal_frac_train:.0%} test — row order)")
        train_idx, test_idx = _temporal_split(df, frac_train=temporal_frac_train)
        fold_iter = [(train_idx, test_idx)]
    else:
        strat_label = df[ev_col].astype(int)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_iter = list(skf.split(df, strat_label))

    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        fold = fold_idx + 1
        n_folds_display = 1 if use_temporal_split else n_folds
        print(f"\n  --- Fold {fold}/{n_folds_display} ---")

        # Scale once per fold — shared across all models on this fold.
        # Fitting _scale_fold inside the model loop would repeat the same
        # StandardScaler fit N×models times with identical output (BUG-M5).
        df_train_full, df_test = _scale_fold(df, train_idx, test_idx, dur_col, ev_col)
        print(f"  Train: {df_train_full.shape[0]} samples | {df_train_full.shape[1] - 2} features "
              f"| event rate={df_train_full[ev_col].gt(0).mean():.3f}", flush=True)
        print(f"  Test:  {df_test.shape[0]} samples | {df_test.shape[1] - 2} features "
              f"| event rate={df_test[ev_col].gt(0).mean():.3f}", flush=True)

        fracs = label_fractions if label_fractions else [1.0]
        for frac in fracs:
            if frac < 1.0:
                n_keep = max(10, int(len(df_train_full) * frac))
                strat = df_train_full[ev_col].astype(int)
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=fold)
                sub_idx, _ = next(sss.split(df_train_full, strat))
                df_train = df_train_full.iloc[sub_idx].reset_index(drop=True)
                frac_tag = f"_frac{frac:.2f}"
            else:
                df_train = df_train_full
                frac_tag = ""

            print(f"  Train ({frac*100:g}%): {df_train.shape[0]} samples | "
                  f"event rate={df_train[ev_col].gt(0).mean():.3f}", flush=True)

            for model_name in model_names:
                model_tag = model_name + frac_tag
                print(f"    {C_BOLD}[Fold {fold}/{n_folds_display}][{model_tag}][{dataset_name}]{C_RESET}", end=" ", flush=True)
                # try:
                run_fold_model(
                    dataset_name, model_name, model_tag,
                    df_train, df_test, dur_col, ev_col,
                    fold, tune, n_trials, output_dir,
                    test_idx=test_idx if save_predictions else None,
                    pred_dir=pred_dir if save_predictions else None,
                    tuned_dir=tuned_dir,
                    **{k: v for k, v in (extra.items() if extra else {}.items())
                        if k != "label_fractions"},
                )
                # except Exception as exc:
                #     warnings.warn(
                #       f"{dataset_name}/{model_tag}/fold_{fold} failed: {exc}",
                #       stacklevel=2,
                #    )
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
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Models or groups to run. Groups: " + ", ".join(MODEL_GROUPS.keys()) + 
             ". Specific models: " + ", ".join(ALL_MODELS.keys()),
    )
    parser.add_argument("--folds",      type=int,   default=5,          help="CV folds.")
    parser.add_argument("--tune",       action="store_true",            help="Enable Optuna tuning.")
    parser.add_argument("--trials",     type=int,   default=10,         help="Optuna trials per fold.")
    parser.add_argument("--output-dir", default="results",              help="Root output directory.")
    parser.add_argument("--seed",       type=int,   default=42,         help="Random seed.")
    parser.add_argument("--num-durations", type=int, default=-1, help="Number of time points to evaluate survival probabilities.")

    g = parser.add_argument_group("TabPFN jointly-trained model options (tabpfn_*)")
    g.add_argument("--epochs",     type=int,   default=50,     help="Training epochs.")
    g.add_argument("--batch-size", type=int,   default=512,     help="Mini-batch size.")
    g.add_argument("--context-size", type=int,   default=512,     help="Context size.")
    g.add_argument("--sampling-ratio", type=float, default=0.5,     help="Ratio of samples to use for training (None = use all).")
    g.add_argument("--n-ensemble", type=int,   default=1,     help="Number of ensemble members.")
    g.add_argument("--lr",         type=float, default=1e-4,   help="Learning rate.")
    g.add_argument("--alpha",      type=float, default=0.5,    help="PFN loss weight.")
    g.add_argument("--device",     type=str,   default="cuda:0", help="Torch device.")
    g.add_argument("--cr-loss-type", type=str, default="deephit", choices=["deephit", "deephit_v2", "cox"], help="Loss type for CR.")

    g3 = parser.add_argument_group("MLHC evaluation options")
    g3.add_argument(
        "--label-fractions", nargs="+", type=float, default=None,
        metavar="F",
        help="Label-efficiency experiment: train on each fraction of training data. "
             "E.g. --label-fractions 0.05 0.1 0.25 0.5 1.0",
    )
    g3.add_argument(
        "--save-predictions", action="store_true",
        help="Save per-fold risk scores for fairness / subgroup evaluation. "
             "Writes <pred-dir>/<dataset>/<model>/fold_N.parquet. "
             "Re-runs any fold whose prediction file is missing even if "
             "metrics.json already exists.",
    )
    g3.add_argument(
        "--pred-dir", default="results/predictions",
        help="Root directory for prediction parquet files "
             "(used with --save-predictions, default: results/predictions).",
    )
    g3.add_argument(
        "--temporal-split", action="store_true",
        help="Replace k-fold CV with a single prospective 70/30 split "
             "(row order assumed to be enrollment/diagnosis order).",
    )
    g3.add_argument(
        "--temporal-frac-train", type=float, default=0.70,
        metavar="F",
        help="Training fraction for --temporal-split (default 0.70).",
    )
    g3.add_argument(
        "--tuned-dir", type=str, default=None,
        help="Path to an existing benchmark results directory. If provided, Optuna "
             "logs are copied from this directory to reuse tuned hyperparameters without retuning.",
    )

    args = parser.parse_args()

    model_names = []
    for m in args.models:
        if m in MODEL_GROUPS:
            model_names.extend(MODEL_GROUPS[m])
        elif m in ALL_MODELS:
            model_names.append(m)
        else:
            parser.error(f"Unknown model or group '{m}'.")

    # Deduplicate while preserving order
    seen = set()
    model_names = [x for x in model_names if not (x in seen or seen.add(x))]

    training_kwargs = {
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "lr":                 args.lr,
        "alpha":              args.alpha,
        "device":             args.device,
        "label_fractions":    args.label_fractions,
        "num_durations":      args.num_durations,
        "cr_loss_type":       args.cr_loss_type,
        "random_state":       args.seed,
        "context_size":       args.context_size,
        "n_ensemble":         args.n_ensemble,
        "sampling_ratio":     args.sampling_ratio,
    }

    logger = LoggerObserver.getLogger("main")
    seed_everything(args.seed)

    for dataset in args.datasets:
        run_benchmark(
            dataset_name=dataset,
            model_names=model_names,
            n_folds=args.folds,
            tune=args.tune,
            n_trials=args.trials,
            output_dir=args.output_dir,
            random_state=args.seed,
            training_kwargs=training_kwargs,
            logger=logger,
            use_temporal_split=args.temporal_split,
            temporal_frac_train=args.temporal_frac_train,
            save_predictions=args.save_predictions,
            pred_dir=args.pred_dir,
            tuned_dir=args.tuned_dir,
        )


if __name__ == "__main__":
    main()
