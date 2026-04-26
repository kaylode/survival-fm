"""plot_calibration.py — Compare raw vs Platt-calibrated risk scores.

Loads saved fold predictions (results/predictions/<dataset>/<model>/fold_*.parquet)
and produces reliability diagrams at one or more time horizons.

Usage
-----
    uv run python -m survpfn.scripts.plot_calibration \\
        --dataset  MIMIC_SURV_B \\
        --model    tabdpt_embedding_mtlr \\
        --results  results/predictions \\
        [--horizons 100 183 335]   # default: Q25/Q50/Q75 of event times (hours)
        [--n-bins  10]             # reliability diagram bins
        [--ipcw]                   # use IPCW weights when fitting Platt
        [--out     calibration.pdf]

The script does leave-one-fold-out Platt scaling: for each held-out fold,
the calibrator is fitted on the remaining folds using the same horizon label,
then applied to the held-out fold.  Raw and calibrated curves are both shown.

NOTE on C-index / IBS
---------------------
Platt scaling is a monotone-in-logit transformation of risk scores.  Because
the transformation is monotone, **C-index (concordance) and time-dependent AUC
are unchanged** by calibration.  Calibration affects only absolute-probability
metrics: Brier Score (lower = better after calibration) and ECE.
IBS improves whenever the curve is better calibrated, but the improvement is
typically small vs ensemble/architecture changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sksurv.metrics import concordance_index_censored


# ── Data loading ────────────────────────────────────────────────────────────

def load_predictions(pred_dir: Path) -> pd.DataFrame:
    """Concatenate all fold_*.parquet files in pred_dir."""
    parts = sorted(pred_dir.glob("fold_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No fold_*.parquet files in {pred_dir}")
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


# ── Horizon label ───────────────────────────────────────────────────────────

def horizon_labels(
    durations: np.ndarray,
    events: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary event-by-tau label and observable mask (no IPCW).

    Returns
    -------
    y_tau : (n,) float — 1 if event occurred by tau, else 0
    observed : (n,) bool — patients whose tau-status is unambiguous
        (event by tau  OR  still at risk at tau)
    """
    is_event = np.asarray(events) > 0
    T = np.asarray(durations)
    y_tau = (is_event & (T <= tau)).astype(float)
    observed = (is_event & (T <= tau)) | (T > tau)
    return y_tau, observed


def km_censoring_at(durations, events, tau, eps=1e-6) -> np.ndarray:
    """IPCW weights = 1 / G(min(T, tau)).  Returns weights array (n,)."""
    T = np.asarray(durations, dtype=float)
    is_event = np.asarray(events) > 0
    c_events = (~is_event).astype(int)          # censoring = event for KM

    order = np.argsort(T)
    T_s, C_s = T[order], c_events[order]

    n, G = len(T_s), 1.0
    at_risk = n
    times_km, G_km = [], []

    for i, (t, c) in enumerate(zip(T_s, C_s)):
        if i > 0 and T_s[i] != T_s[i - 1]:
            times_km.append(T_s[i - 1])
            G_km.append(G)
        if c == 1:
            G *= 1.0 - 1.0 / max(at_risk, 1)
        at_risk -= 1
    times_km.append(T_s[-1])
    G_km.append(G)
    times_km, G_km = np.array(times_km), np.array(G_km)

    t_eval = np.minimum(T, tau)
    idx = np.searchsorted(times_km, t_eval, side="right") - 1
    idx = np.clip(idx, 0, len(G_km) - 1)
    G_vals = np.maximum(G_km[idx], eps)

    y_tau, observed = horizon_labels(durations, events, tau)
    weights = np.where(observed, 1.0 / G_vals, 0.0)
    return weights


# ── Leave-one-fold-out Platt calibration ────────────────────────────────────

def lovo_platt_calibrate(
    risk: np.ndarray,
    y_tau: np.ndarray,
    observed: np.ndarray,
    folds: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Leave-one-fold-out Platt calibration.

    For each held-out fold, fit a logistic regression on logit(risk) using the
    remaining folds, then predict calibrated probabilities for the held-out fold.

    Returns calibrated risk array (same shape as risk).
    """
    eps = 1e-6
    risk_cal = risk.copy()

    for test_fold in np.unique(folds):
        is_test = folds == test_fold
        is_train = ~is_test & observed

        if is_train.sum() < 10 or len(np.unique(y_tau[is_train])) < 2:
            continue  # not enough data or only one class

        x_tr = logit(np.clip(risk[is_train], eps, 1 - eps)).reshape(-1, 1)
        y_tr = y_tau[is_train]
        w_tr = weights[is_train] if weights is not None else None

        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        platt.fit(x_tr, y_tr, sample_weight=w_tr)

        x_te = logit(np.clip(risk[is_test], eps, 1 - eps)).reshape(-1, 1)
        risk_cal[is_test] = platt.predict_proba(x_te)[:, 1]

    return risk_cal


# ── Expected Calibration Error ───────────────────────────────────────────────

def ece(frac_pos: np.ndarray, mean_pred: np.ndarray, counts: np.ndarray) -> float:
    """Weighted ECE from calibration_curve outputs."""
    n = counts.sum()
    if n == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(frac_pos - mean_pred)) / n)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _compute_metrics(
    r: np.ndarray,
    y_obs: np.ndarray,
    T_obs: np.ndarray,
    events_obs: np.ndarray,
    tau: float,
    n_bins: int,
) -> dict:
    """Return a dict of ECE, BS, AUC, C_td for a single risk array."""
    try:
        frac_pos, mean_pred = calibration_curve(y_obs, r, n_bins=n_bins, strategy="quantile")
    except TypeError:
        frac_pos, mean_pred = calibration_curve(y_obs, r, n_bins=n_bins)

    bin_edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    bin_ids   = np.digitize(r, bin_edges[1:-1])
    counts    = np.array([(bin_ids == b).sum() for b in range(n_bins)])
    ece_val   = ece(frac_pos, mean_pred, counts[: len(frac_pos)])

    auc_val  = roc_auc_score(y_obs, r)
    bs_val   = brier_score_loss(y_obs, r)

    trunc_T  = np.minimum(T_obs, tau)
    trunc_E  = (events_obs > 0) & (T_obs <= tau)
    c_td_val, _, _, _, _ = concordance_index_censored(trunc_E.astype(bool), trunc_T, r)

    return {
        "ece":  ece_val,
        "bs":   bs_val,
        "auc":  auc_val,
        "c_td": c_td_val,
        "frac_pos":  frac_pos,
        "mean_pred": mean_pred,
    }


def plot_calibration_comparison(
    df: pd.DataFrame,
    horizons: list[float],
    n_bins: int = 10,
    use_ipcw: bool = False,
    title: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Reliability diagrams for raw vs calibrated at each horizon.

    Parameters
    ----------
    df : DataFrame with columns [risk_score, duration, event, fold]
    horizons : list of time horizons
    n_bins : reliability diagram bins
    use_ipcw : weight Platt fitting by IPCW
    title : figure suptitle

    Returns
    -------
    fig : matplotlib Figure
    summary : DataFrame — one row per (horizon, variant) with ECE/BS/AUC/C_td
    """
    n_h = len(horizons)
    fig, axes = plt.subplots(
        n_h, 2,
        figsize=(10, 4 * n_h),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    risk   = df["risk_score"].values
    T      = df["duration"].values
    events = df["event"].values
    folds  = df["fold"].values

    records = []

    for row, tau in enumerate(horizons):
        y_tau, observed = horizon_labels(T, events, tau)
        ipcw_w = km_censoring_at(T, events, tau) if use_ipcw else None

        # Calibrated via leave-one-fold-out Platt
        risk_cal = lovo_platt_calibrate(risk, y_tau, observed, folds, weights=ipcw_w)

        # Use only observable patients for the reliability diagram
        obs_idx      = np.where(observed)[0]
        y_obs        = y_tau[obs_idx]
        risk_obs     = np.clip(risk[obs_idx],     0.0, 1.0)
        risk_cal_obs = np.clip(risk_cal[obs_idx], 0.0, 1.0)

        T_obs      = T[obs_idx]
        events_obs = events[obs_idx]

        tau_label = f"τ = {tau:.1f}"
        obs_rate  = y_obs.mean()

        for col, (r, variant, color) in enumerate([
            (risk_obs,     "Raw",              "#4C72B0"),
            (risk_cal_obs, "Calibrated(Platt)", "#DD8452"),
        ]):
            ax = axes[row][col]
            m  = _compute_metrics(r, y_obs, T_obs, events_obs, tau, n_bins)

            records.append({
                "horizon":   tau,
                "variant":   variant,
                "n_obs":     len(obs_idx),
                "prevalence": float(obs_rate),
                "ECE":       float(m["ece"]),
                "BS":        float(m["bs"]),
                "AUC":       float(m["auc"]),
                "C_td":      float(m["c_td"]),
            })

            legend_body = (
                f"{variant}\n"
                f"ECE={m['ece']:.3f}  BS={m['bs']:.3f}\n"
                f"AUC={m['auc']:.3f}  C_td={m['c_td']:.3f}"
            )

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
            ax.plot(m["mean_pred"], m["frac_pos"], "o-", color=color, lw=2,
                    markersize=6, label=legend_body)
            ax.axhline(obs_rate, color="gray", ls=":", lw=1,
                       label=f"Prevalence={obs_rate:.3f}")

            # Annotate each panel with a clean metric table in the lower-right
            metric_str = (
                f"ECE  = {m['ece']:.4f}\n"
                f"BS   = {m['bs']:.4f}\n"
                f"AUC  = {m['auc']:.4f}\n"
                f"C_td = {m['c_td']:.4f}"
            )
            ax.text(
                0.98, 0.05, metric_str,
                transform=ax.transAxes,
                fontsize=8.5, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontfamily="monospace",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted risk", fontsize=11)
            ax.set_ylabel("Observed event fraction", fontsize=11)
            ax.set_title(f"{tau_label}  |  {variant}", fontsize=11)
            ax.legend(fontsize=8.5, loc="upper left")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    summary = pd.DataFrame(records)
    return fig, summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Calibration comparison plot")
    p.add_argument("--dataset",  required=True, help="e.g. MIMIC_SURV_B")
    p.add_argument("--model",    required=True, help="e.g. tabdpt_embedding_mtlr")
    p.add_argument("--results",  default="results/predictions",
                   help="Root predictions directory (default: results/predictions)")
    p.add_argument("--horizons", nargs="+", type=float, default=None,
                   help="Time horizons in data units (default: Q25/Q50/Q75 of event times)")
    p.add_argument("--n-bins",   type=int, default=10,
                   help="Number of reliability diagram bins (default: 10)")
    p.add_argument("--ipcw",     action="store_true",
                   help="Use IPCW weights when fitting Platt calibrator")
    p.add_argument("--out",      default=None,
                   help="Output file (default: <dataset>_<model>_calibration.pdf)")
    return p.parse_args()


def main():
    args = parse_args()

    pred_dir = Path(args.results) / args.dataset / args.model
    df = load_predictions(pred_dir)
    print(f"Loaded {len(df)} predictions  "
          f"({(df['event'] > 0).sum()} events, "
          f"{df['fold'].nunique()} folds)")

    # Default horizons: Q25/Q50/Q75 of event times
    if args.horizons is None:
        et = df.loc[df["event"] > 0, "duration"]
        horizons = np.quantile(et, [0.25, 0.50, 0.75]).tolist()
        print(f"Auto horizons (Q25/Q50/Q75 of event times): "
              + ", ".join(f"{h:.1f}" for h in horizons))
    else:
        horizons = args.horizons

    out_path = args.out or f"{args.dataset}_{args.model}_calibration.pdf"

    title = f"{args.dataset} / {args.model}"
    if args.ipcw:
        title += "  [IPCW-weighted Platt]"

    fig, summary = plot_calibration_comparison(
        df,
        horizons=horizons,
        n_bins=args.n_bins,
        use_ipcw=args.ipcw,
        title=title,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot → {out_path}")

    # Save numeric summary as CSV next to the plot
    csv_path = str(out_path).rsplit(".", 1)[0] + "_metrics.csv"
    summary.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"Saved metrics → {csv_path}")

    # Print comparison table to stdout
    print("\n── Calibration metric comparison ──────────────────────────────")
    print(f"{'Horizon':>10}  {'Variant':<22}  {'ECE':>7}  {'BS':>7}  {'AUC':>7}  {'C_td':>7}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(
            f"{row['horizon']:>10.1f}  {row['variant']:<22}  "
            f"{row['ECE']:>7.4f}  {row['BS']:>7.4f}  "
            f"{row['AUC']:>7.4f}  {row['C_td']:>7.4f}"
        )
    print()
    print("NOTE: C-index/AUC are rank-based → unchanged by Platt calibration.")
    print("      ECE and BS improve when predictions are better calibrated.")


if __name__ == "__main__":
    main()
