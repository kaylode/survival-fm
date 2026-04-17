"""
survpfn/xai/analyze_data_comprehensive.py — Per-dataset EDA for SurvPFN.

For each dataset in the selected group, generates:
  {name}_km.pdf            — Kaplan–Meier survival curve with 95% CI
  {name}_time_dist.pdf     — event-time histogram split by event / censored
  {name}_feature_corr.pdf  — top-20 features by |correlation with event|
  {name}_missing.pdf       — missing-value bar chart (skipped if no missing)

Cross-dataset summary (written to <output-dir>/summary/):
  km_overlay.pdf           — KM curves for all datasets on one plot (≤12 datasets)
  followup_violin.pdf      — follow-up time distributions per dataset

Usage
-----
    uv run python -m survpfn.xai.analyze_data_comprehensive --group public
    uv run python -m survpfn.xai.analyze_data_comprehensive --group ehr
    uv run python -m survpfn.xai.analyze_data_comprehensive --group public+ehr
    uv run python -m survpfn.xai.analyze_data_comprehensive --group survset
    uv run python -m survpfn.xai.analyze_data_comprehensive --group all
    uv run python -m survpfn.xai.analyze_data_comprehensive --datasets VETERANS WHAS500
    uv run python -m survpfn.xai.analyze_data_comprehensive --group public --no-summary
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    from lifelines import KaplanMeierFitter
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False
    warnings.warn(
        "lifelines not installed — KM curves will be skipped. "
        "Install with: uv add lifelines",
        stacklevel=1,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# Colour cycle for multi-line plots
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
           "#aec7e8", "#ffbb78"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_km_curve(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    name: str,
    out_path: Path,
) -> None:
    """Kaplan–Meier survival curve with 95 % CI and censoring tick marks."""
    if not _HAS_LIFELINES:
        return

    event = (df[ev_col] > 0).astype(int)
    duration = df[dur_col].astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event, label=f"{name} (N={len(df):,})")
    kmf.plot_survival_function(
        ax=ax, ci_show=True, ci_alpha=0.15, color="#1f77b4",
        at_risk_counts=False,
    )

    # Annotate median survival if it exists
    med = kmf.median_survival_time_
    if np.isfinite(med):
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(med, color="gray", linestyle=":", linewidth=0.8)
        ax.annotate(
            f"med = {med:.1f}",
            (med, 0.5), textcoords="offset points", xytext=(4, 4),
            fontsize=7, color="gray",
        )

    n_events = int(event.sum())
    n_cens   = int((1 - event).sum())
    ax.set_title(
        f"{name} — KM curve\n"
        f"events={n_events:,}  censored={n_cens:,}  "
        f"event rate={event.mean():.2f}",
        fontsize=9,
    )
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel("Survival probability", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_distribution(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    name: str,
    out_path: Path,
) -> None:
    """
    Stacked histogram of event times vs censoring times.
    Provides a visual of the hazard profile and observation window.
    """
    event = (df[ev_col] > 0).astype(int)
    duration = df[dur_col].astype(float)

    ev_times   = duration[event == 1].values
    cens_times = duration[event == 0].values

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

    n_bins = min(50, max(10, int(np.sqrt(len(duration)))))

    # Event times
    if len(ev_times):
        axes[0].hist(ev_times, bins=n_bins, color="#d62728", alpha=0.75,
                     edgecolor="white", linewidth=0.4)
        axes[0].axvline(np.median(ev_times), color="black", linestyle="--",
                        linewidth=1, label=f"median={np.median(ev_times):.1f}")
        axes[0].legend(fontsize=7)
    axes[0].set_title(f"Event times (n={len(ev_times):,})", fontsize=9)
    axes[0].set_xlabel("Time", fontsize=9)
    axes[0].set_ylabel("Count", fontsize=9)

    # Censoring times
    if len(cens_times):
        axes[1].hist(cens_times, bins=n_bins, color="#1f77b4", alpha=0.75,
                     edgecolor="white", linewidth=0.4)
        axes[1].axvline(np.median(cens_times), color="black", linestyle="--",
                        linewidth=1, label=f"median={np.median(cens_times):.1f}")
        axes[1].legend(fontsize=7)
    axes[1].set_title(f"Censoring times (n={len(cens_times):,})", fontsize=9)
    axes[1].set_xlabel("Time", fontsize=9)

    fig.suptitle(f"{name} — follow-up time distribution", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    name: str,
    out_path: Path,
    top_n: int = 25,
) -> None:
    """
    Horizontal bar chart of the top-N features by |correlation with event|.
    Uses point-biserial r for continuous features, Cramér's V for binary/categorical.
    """
    event = (df[ev_col] > 0).astype(float)
    X = df.drop(columns=[dur_col, ev_col], errors="ignore")

    records = []
    for col in X.columns:
        s = X[col].dropna()
        idx = s.index.intersection(event.index)
        e = event.loc[idx]
        s = s.loc[idx]
        if len(s) < 10:
            continue
        if s.nunique() <= 10:
            ct = pd.crosstab(s, e)
            chi2 = scipy_stats.chi2_contingency(ct, correction=False)[0]
            n = ct.values.sum()
            r, c = ct.shape
            denom = n * (min(r, c) - 1)
            score = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
            stype = "Cramér's V"
        else:
            r, _ = scipy_stats.pointbiserialr(e.astype(float), s.astype(float))
            score = float(r)
            stype = "point-biserial r"
        records.append({"feature": col, "score": score,
                         "abs_score": abs(score), "type": stype})

    if not records:
        return

    fdf = (pd.DataFrame(records)
           .sort_values("abs_score", ascending=False)
           .head(top_n))

    fig, ax = plt.subplots(figsize=(8, max(3, len(fdf) * 0.32)))
    colors = ["#d62728" if t == "point-biserial r" else "#1f77b4"
              for t in fdf["type"]]
    y = np.arange(len(fdf))
    ax.barh(y, fdf["abs_score"].values[::-1], 0.7, color=colors[::-1],
            alpha=0.82, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(fdf["feature"].values[::-1],
                       fontsize=max(5, 8 - len(fdf) // 10))
    ax.set_xlabel("|correlation with event|", fontsize=9)
    ax.set_title(f"{name} — top {top_n} feature–event correlations", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#d62728", label="point-biserial r"),
                 Patch(color="#1f77b4", label="Cramér's V")],
        fontsize=7, loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missing_bars(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    name: str,
    out_path: Path,
) -> None:
    """
    Horizontal bar chart of missing data per feature.
    Only saved if at least one feature has missing values.
    """
    X = df.drop(columns=[dur_col, ev_col], errors="ignore")
    miss = (X.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(miss) * 0.28)))
    y = np.arange(len(miss))
    ax.barh(y, miss.values[::-1], 0.7, color="#ff7f0e", alpha=0.82,
            edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(miss.index[::-1],
                       fontsize=max(5, 8 - len(miss) // 12))
    ax.set_xlabel("Missing %", fontsize=9)
    ax.set_title(f"{name} — missing values per feature", fontsize=9)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_km_overlay(
    records: list[dict],
    out_path: Path,
) -> None:
    """
    All KM curves overlaid on one plot.  Suitable for ≤12 datasets.
    """
    if not _HAS_LIFELINES or not records:
        return

    n = len(records)
    if n > 12:
        # Too crowded; skip
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, rec in enumerate(records):
        color = _COLORS[i % len(_COLORS)]
        kmf = KaplanMeierFitter()
        event = (rec["df"][rec["ev_col"]] > 0).astype(int)
        dur   = rec["df"][rec["dur_col"]].astype(float)
        # Normalise time to [0,1] for comparability
        dur_n = dur / dur.max() if dur.max() > 0 else dur
        kmf.fit(dur_n, event_observed=event, label=rec["name"])
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=1.5)

    ax.set_xlabel("Normalised time (0 = start, 1 = max follow-up)", fontsize=9)
    ax.set_ylabel("Survival probability", fontsize=9)
    ax.set_title("KM curves — all datasets (time normalised)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, ncol=max(1, n // 6), loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_followup_violin(
    records: list[dict],
    out_path: Path,
) -> None:
    """
    Violin plot of normalised follow-up time per dataset.
    Helps show heterogeneity in observation window shape.
    """
    if not records:
        return

    data   = []
    labels = []
    for rec in records:
        dur = rec["df"][rec["dur_col"]].astype(float).values
        if dur.max() > 0:
            dur = dur / dur.max()
        data.append(dur)
        labels.append(rec["name"])

    fig, ax = plt.subplots(figsize=(max(6, len(records) * 0.9), 4))
    parts = ax.violinplot(data, positions=np.arange(len(data)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_alpha(0.55)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       fontsize=max(6, 9 - len(labels) // 5))
    ax.set_ylabel("Normalised follow-up time", fontsize=9)
    ax.set_title("Follow-up time distribution per dataset", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def analyse_dataset(
    name: str,
    out_dir: Path,
) -> dict | None:
    """Run all EDA plots for one dataset. Returns a record dict for summary."""
    from survpfn.dataloaders import ALL_DATASETS

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")

    loader = ALL_DATASETS.get(name)
    if loader is None:
        print(f"  [SKIP] not in registry")
        return None

    try:
        df, dur_col, ev_col = loader()
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return None

    n      = len(df)
    p      = len(df.columns) - 2
    ev_r   = (df[ev_col] > 0).mean()
    print(f"  N={n:,}  features={p}  event_rate={ev_r:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # KM curve
    km_path = out_dir / f"{name}_km.pdf"
    print(f"  → KM curve ...")
    plot_km_curve(df, dur_col, ev_col, name, km_path)

    # Time distribution
    td_path = out_dir / f"{name}_time_dist.pdf"
    print(f"  → time distribution ...")
    plot_time_distribution(df, dur_col, ev_col, name, td_path)

    # Feature–event correlation
    fc_path = out_dir / f"{name}_feature_corr.pdf"
    print(f"  → feature correlation ...")
    plot_feature_correlation(df, dur_col, ev_col, name, fc_path)

    # Missing values
    mv_path = out_dir / f"{name}_missing.pdf"
    print(f"  → missing values ...")
    plot_missing_bars(df, dur_col, ev_col, name, mv_path)

    return {"name": name, "df": df, "dur_col": dur_col, "ev_col": ev_col}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from survpfn.dataloaders import DATASET_GROUPS

    parser = argparse.ArgumentParser(
        description="Per-dataset EDA figures for SurvPFN."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--group", choices=list(DATASET_GROUPS), default=None,
        help="Predefined dataset group.",
    )
    grp.add_argument(
        "--datasets", nargs="+", default=None,
        help="Explicit list of dataset names.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/xai/datasets/eda"),
        help="Root output directory (group sub-folder created automatically).",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip cross-dataset summary figures.",
    )
    args = parser.parse_args()

    # Resolve dataset names
    if args.datasets:
        names  = args.datasets
        subdir = "custom"
    elif args.group:
        names  = DATASET_GROUPS[args.group]
        subdir = args.group
    else:
        names  = DATASET_GROUPS["public"]
        subdir = "public"

    out_root = args.output_dir / subdir
    print(f"\nEDA — {len(names)} dataset(s) → {out_root}")

    records = []
    for name in names:
        rec = analyse_dataset(name, out_root / name)
        if rec is not None:
            records.append(rec)

    # Cross-dataset summary
    if not args.no_summary and records:
        summary_dir = out_root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating summary figures → {summary_dir}")
        plot_km_overlay(records,       summary_dir / "km_overlay.pdf")
        plot_followup_violin(records,  summary_dir / "followup_violin.pdf")

    print(f"\nDone — {len(records)}/{len(names)} datasets processed.")


if __name__ == "__main__":
    main()
