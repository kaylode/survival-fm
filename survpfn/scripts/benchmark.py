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
  survtrace, deepsurv, mtlr, pchazard, deephit_single
  tabpfn_embedding_cox, tabpfn_embedding_deephit, tabpfn_embedding_pchazard, tabpfn_embedding_mtlr
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
from survpfn.metrics import evaluate_sr, evaluate_cr
from survpfn.utils.config import seed_everything
from survpfn.utils.logger import LoggerObserver
from survpfn.utils.optuna import (
    _scale_fold,
    get_feature_importance,
    get_best_params,
    count_params
)


# ---------------------------------------------------------------------------
# Model groups for CLI selection
# ---------------------------------------------------------------------------

MODEL_GROUPS: dict[str, list[str]] = {
    "classical": ["cox", "km", "rsf", "gbsa"],
    "deep":      ["deepsurv", "mtlr", "pchazard", "deephit_single", "survtrace", "soden", "beta_surv"],

    "tabpfn_embedding": [
        "tabpfn_embedding_cox", "tabpfn_embedding_cox_adapter",
        "tabpfn_embedding_deephit", "tabpfn_embedding_deephit_adapter",
        "tabpfn_embedding_pchazard", "tabpfn_embedding_pchazard_adapter",
        "tabpfn_embedding_mtlr", "tabpfn_embedding_mtlr_adapter"
    ],
    "tabdpt_embedding": [
        "tabdpt_embedding_cox", "tabdpt_embedding_cox_adapter",
        "tabdpt_embedding_deephit", "tabdpt_embedding_deephit_adapter",
        "tabdpt_embedding_pchazard", "tabdpt_embedding_pchazard_adapter",
        "tabdpt_embedding_mtlr", "tabdpt_embedding_mtlr_adapter"
    ],
    "tabicl_embedding": [
        "tabicl_embedding_cox", "tabicl_embedding_cox_adapter",
        "tabicl_embedding_deephit", "tabicl_embedding_deephit_adapter",
        "tabicl_embedding_pchazard", "tabicl_embedding_pchazard_adapter",
        "tabicl_embedding_mtlr", "tabicl_embedding_mtlr_adapter"
    ],
    "tabpfn_joint": [
        "tabpfn_joint_cox", "tabpfn_joint_cox_adapter",
        "tabpfn_joint_deephit", "tabpfn_joint_deephit_adapter",
        "tabpfn_joint_pchazard", "tabpfn_joint_pchazard_adapter",
        "tabpfn_joint_mtlr", "tabpfn_joint_mtlr_adapter"
    ],
    "tabdpt_joint": [
        "tabdpt_joint_cox", "tabdpt_joint_cox_adapter",
        "tabdpt_joint_deephit", "tabdpt_joint_deephit_adapter",
        "tabdpt_joint_pchazard", "tabdpt_joint_pchazard_adapter",
        "tabdpt_joint_mtlr", "tabdpt_joint_mtlr_adapter"
    ],
    "tabicl_joint": [
        "tabicl_joint_cox", "tabicl_joint_cox_adapter",
        "tabicl_joint_deephit", "tabicl_joint_deephit_adapter",
        "tabicl_joint_pchazard", "tabicl_joint_pchazard_adapter",
        "tabicl_joint_mtlr", "tabicl_joint_mtlr_adapter"
    ],

    "zeroshot": ["tabpfn_zeroshot", "tabdpt_zeroshot", "tabicl_zeroshot"],

    "cr": [
        "cox_cr", "deephit_cr",
        "tabpfn_embedding_deephit_cr", "tabpfn_embedding_deephit_v2_cr", "tabpfn_embedding_deephit_v2_cr_adapter",
        "tabpfn_embedding_cox_cr", "tabpfn_embedding_cox_cr_adapter",
        "tabdpt_embedding_deephit_cr", "tabdpt_embedding_deephit_v2_cr", "tabdpt_embedding_deephit_v2_cr_adapter",
        "tabdpt_embedding_cox_cr", "tabdpt_embedding_cox_cr_adapter",
        "tabicl_embedding_deephit_cr", "tabicl_embedding_deephit_v2_cr", "tabicl_embedding_deephit_v2_cr_adapter",
        "tabicl_embedding_cox_cr", "tabicl_embedding_cox_cr_adapter",
        "tabpfn_joint_deephit_cr", "tabpfn_joint_deephit_v2_cr", "tabpfn_joint_deephit_v2_cr_adapter",
        "tabpfn_joint_cox_cr", "tabpfn_joint_cox_cr_adapter",
        "tabdpt_joint_deephit_cr", "tabdpt_joint_deephit_v2_cr", "tabdpt_joint_deephit_v2_cr_adapter",
        "tabdpt_joint_cox_cr", "tabdpt_joint_cox_cr_adapter",
        "tabicl_joint_deephit_cr", "tabicl_joint_deephit_v2_cr", "tabicl_joint_deephit_v2_cr_adapter",
        "tabicl_joint_cox_cr", "tabicl_joint_cox_cr_adapter"
    ],
}

# Aggregate groups
MODEL_GROUPS["fm_embedding"] = (
    MODEL_GROUPS["tabpfn_embedding"] +
    MODEL_GROUPS["tabdpt_embedding"] +
    MODEL_GROUPS["tabicl_embedding"]
)
MODEL_GROUPS["fm_joint"] = (
    MODEL_GROUPS["tabpfn_joint"] +
    MODEL_GROUPS["tabdpt_joint"] +
    MODEL_GROUPS["tabicl_joint"]
)
MODEL_GROUPS["tabpfn"]       = MODEL_GROUPS["tabpfn_embedding"] + MODEL_GROUPS["tabpfn_joint"]
MODEL_GROUPS["tabdpt"]       = MODEL_GROUPS["tabdpt_embedding"] + MODEL_GROUPS["tabdpt_joint"]
MODEL_GROUPS["tabicl"]       = MODEL_GROUPS["tabicl_embedding"] + MODEL_GROUPS["tabicl_joint"]
MODEL_GROUPS["fm"]           = (
    MODEL_GROUPS["fm_embedding"] + MODEL_GROUPS["fm_joint"] + MODEL_GROUPS["zeroshot"]
)

# Default 'all' (Single Risk)
MODEL_GROUPS["all"] = (
    MODEL_GROUPS["classical"] +
    MODEL_GROUPS["deep"] +
    MODEL_GROUPS["fm_embedding"] +
    MODEL_GROUPS["fm_joint"] +
    MODEL_GROUPS["zeroshot"]
)


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
    **training_kwargs,
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
    
    # ── Skip check ────────────────────────────────────────────────────────
    metrics_file = os.path.join(out_dir, "metrics.json")
    params_file = os.path.join(out_dir, "best_params.json")
    if os.path.exists(metrics_file) and (not tune or os.path.exists(params_file)):
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            c = metrics.get("C-index", float("nan"))
            ibs = metrics.get("IBS", float("nan"))
            auc = metrics.get("AUC_mean", float("nan"))
            if not np.isnan(c):
                print(f"      {C_BLUE}→ SKIPPING (C={c:.4f} IBS={ibs:.4f} AUC={auc:.4f}){C_RESET}")
                return metrics
            else:
                print(f"      {C_YELLOW}→ Metric is NaN, retraining{C_RESET}")
        except Exception:
            pass

    os.makedirs(out_dir, exist_ok=True)

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

    # ── Evaluate metrics ────────────────────────────────────────────────────
    t_eval = time.perf_counter()
    num_events = int(df_train[event_col].max())
    if num_events > 1:
        # Competing risks: surv_p is list of CIFs
        metrics = evaluate_cr(
            df_train, df_test, duration_col, event_col,
            cif_per_cause=surv_p, surv_times=surv_t,
        )
    else:
        # Single risk: surv_p is matrix of S(t)
        metrics = evaluate_sr(
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
    ibs = metrics.get("IBS", float("nan"))
    auc = metrics.get("AUC_mean", float("nan"))
    t_total = fit_time_s + eval_time_s
    if not np.isnan(c):
        print(f"      {C_GREEN}→ C={c:.4f} IBS={ibs:.4f} AUC={auc:.4f}{C_RESET}  fit={fit_time_s:.1f}s")
    else:
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
    training_kwargs:
        Extra args forwarded to jointly-trained TabPFN models (``tabpfn_*``):
        ``epochs``, ``batch_size``, ``lr``, ``alpha``, ``device``.
    """
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    df, dur_col, ev_col = get_dataset(dataset_name)
    print(f"  Loaded: {df.shape[0]} samples | {df.shape[1] - 2} features "
          f"| event rate={df[ev_col].gt(0).mean():.3f}")

    # For competing-risks datasets (event ∈ {0,1,2,…}) stratify on the full
    # cause code so each fold receives a proportional share of every cause.
    # For single-event datasets the cause code IS the binary event flag.
    strat_label = df[ev_col].astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    extra = training_kwargs or {}

    label_fractions = (training_kwargs or {}).get("label_fractions", None)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, strat_label)):
        fold = fold_idx + 1
        print(f"\n  --- Fold {fold}/{n_folds} ---")

        for model_name in model_names:
            df_train_full, df_test = _scale_fold(
                df, train_idx, test_idx, dur_col, ev_col
            )

            # Label-efficiency loop: subsample training set to each fraction
            fracs = label_fractions if label_fractions else [1.0]
            for frac in fracs:
                if frac < 1.0:
                    n_keep = max(10, int(len(df_train_full) * frac))
                    # Stratified subsample — use full cause code for CR datasets
                    strat = df_train_full[ev_col].astype(int)
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=fold)
                    sub_idx, _ = next(sss.split(df_train_full, strat))
                    df_train = df_train_full.iloc[sub_idx].reset_index(drop=True)
                    frac_tag = f"_frac{frac:.2f}"
                else:
                    df_train = df_train_full
                    frac_tag = ""

                model_tag = model_name + frac_tag
                print(f"    {C_BOLD}[{model_tag}][{dataset_name}]{C_RESET}", end=" ", flush=True)
                # try:
                run_fold_model(
                    dataset_name, model_tag,
                    df_train, df_test, dur_col, ev_col,
                    fold, tune, n_trials, output_dir,
                    **{k: v for k, v in (extra.items() if extra else {}.items())
                       if k != "label_fractions"},
                )
                # except Exception as exc:
                #     warnings.warn(
                #         f"{dataset_name}/{model_tag}/fold_{fold} failed: {exc}",
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
    parser.add_argument("--num-durations", type=int, default=10, help="Number of time points to evaluate survival probabilities.")

    g = parser.add_argument_group("TabPFN jointly-trained model options (tabpfn_*)")
    g.add_argument("--epochs",     type=int,   default=50,     help="Training epochs.")
    g.add_argument("--batch-size", type=int,   default=64,     help="Mini-batch size.")
    g.add_argument("--lr",         type=float, default=1e-3,   help="Learning rate.")
    g.add_argument("--alpha",      type=float, default=1.0,    help="PFN loss weight.")
    g.add_argument("--device",     type=str,   default="cuda:0", help="Torch device.")
    g.add_argument("--cr-loss-type", type=str, default="deephit", choices=["deephit", "deephit_v2", "cox"], help="Loss type for CR.")

    g3 = parser.add_argument_group("MLHC evaluation options")
    g3.add_argument(
        "--label-fractions", nargs="+", type=float, default=None,
        metavar="F",
        help="Label-efficiency experiment: train on each fraction of training data. "
             "E.g. --label-fractions 0.05 0.1 0.25 0.5 1.0",
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
            logger=logger
        )


if __name__ == "__main__":
    main()
