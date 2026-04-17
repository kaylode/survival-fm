"""
survpfn/xai/correlation_analysis.py — Dataset quality audit for SurvPFN.

Detects three classes of data problems before modelling:

  1. **Label leakage** — features suspiciously correlated with the event
     indicator (point-biserial r or Cramér's V > threshold).

  2. **Complete / quasi-complete separation** — a feature has near-zero
     variance *within* one event stratum (triggers lifelines ConvergenceWarning).

  3. **Inter-feature multicollinearity** — pairs of features with |r| > 0.95
     that may destabilise Cox / linear heads.

Outputs (per dataset, saved under ``SAVE_DIR``):
  • ``<DATASET>_leakage.csv``        — leakage scores for all features
  • ``<DATASET>_separation.csv``     — separation flags + stratum variances
  • ``<DATASET>_collinear_pairs.csv``— highly correlated feature pairs
  • ``<DATASET>_corr_heatmap.png``   — full feature correlation heatmap
  • ``summary_flags.csv``            — one row per (dataset, feature) with all flags

Usage
-----
    # All registered datasets
    uv run python -m survpfn.xai.correlation_analysis

    # Specific datasets
    uv run python -m survpfn.xai.correlation_analysis --datasets VETERANS WHAS500 METABRIC

    # Adjust thresholds
    uv run python -m survpfn.xai.correlation_analysis \\
        --leakage-threshold 0.7 \\
        --separation-var-threshold 1e-4 \\
        --collinear-threshold 0.95
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SAVE_DIR = Path("results/xai/datasets/correlation")

# Thresholds (overridable via CLI)
LEAKAGE_THRESHOLD    = 0.70   # |correlation with event| above this = leakage risk
SEPARATION_VAR_THR   = 1e-3   # variance within event stratum below this = separation
COLLINEAR_THRESHOLD  = 0.95   # |pairwise r| above this = multicollinearity flag
MAX_HEATMAP_FEATURES = 60     # cap heatmap size for readability


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _point_biserial(feature: pd.Series, event: pd.Series) -> float:
    """Pearson r between a continuous feature and binary event indicator."""
    mask = feature.notna() & event.notna()
    if mask.sum() < 10:
        return np.nan
    r, _ = stats.pointbiserialr(event[mask].astype(float), feature[mask])
    return float(r)


def _cramers_v(feature: pd.Series, event: pd.Series) -> float:
    """Cramér's V between a categorical/binary feature and event indicator."""
    mask = feature.notna() & event.notna()
    if mask.sum() < 10:
        return np.nan
    ct = pd.crosstab(feature[mask], event[mask])
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, c = ct.shape
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


def _is_binary_or_categorical(s: pd.Series, max_unique: int = 10) -> bool:
    return s.nunique() <= max_unique


# ---------------------------------------------------------------------------
# Per-dataset analyses
# ---------------------------------------------------------------------------

def leakage_analysis(
    X: pd.DataFrame,
    event: pd.Series,
    threshold: float = LEAKAGE_THRESHOLD,
) -> pd.DataFrame:
    """Compute correlation of every feature with the event indicator.

    Returns a DataFrame sorted by |score| descending with columns:
        feature, score_type, score, flagged
    """
    records = []
    for col in X.columns:
        s = X[col]
        if _is_binary_or_categorical(s):
            score = _cramers_v(s, event)
            stype = "cramers_v"
        else:
            score = _point_biserial(s, event)
            stype = "point_biserial_r"
        records.append({
            "feature":    col,
            "score_type": stype,
            "score":      score,
            "abs_score":  abs(score) if score is not None and not np.isnan(score) else np.nan,
            "flagged":    (not np.isnan(score)) and abs(score) >= threshold,
        })
    df = pd.DataFrame(records).sort_values("abs_score", ascending=False).reset_index(drop=True)
    return df


def separation_analysis(
    X: pd.DataFrame,
    event: pd.Series,
    var_threshold: float = SEPARATION_VAR_THR,
) -> pd.DataFrame:
    """Detect complete / quasi-complete separation.

    For each feature computes variance within event=1 and event=0 strata.
    A feature is flagged when either stratum variance is below ``var_threshold``.
    This directly reproduces the lifelines ConvergenceWarning check.

    Returns DataFrame with columns:
        feature, var_event1, var_event0, min_var, flagged, note
    """
    ev1 = event.astype(bool)
    records = []
    for col in X.columns:
        s = X[col]
        v1 = float(s[ev1].var())   if ev1.sum()  > 1 else np.nan
        v0 = float(s[~ev1].var())  if (~ev1).sum() > 1 else np.nan
        min_v = np.nanmin([v1, v0])
        flagged = (not np.isnan(min_v)) and min_v < var_threshold
        note = ""
        if flagged:
            stratum = "event=1" if (not np.isnan(v1) and v1 < var_threshold) else "event=0"
            note = f"near-zero variance in {stratum} stratum → possible complete separation"
        records.append({
            "feature":   col,
            "var_event1": v1,
            "var_event0": v0,
            "min_var":    min_v,
            "flagged":    flagged,
            "note":       note,
        })
    df = pd.DataFrame(records).sort_values("min_var", ascending=True).reset_index(drop=True)
    return df


def collinearity_analysis(
    X: pd.DataFrame,
    threshold: float = COLLINEAR_THRESHOLD,
) -> pd.DataFrame:
    """Find pairs of features with |Pearson r| >= threshold.

    Returns DataFrame with columns:
        feature_a, feature_b, pearson_r, abs_r
    Sorted by abs_r descending.
    """
    # Drop constant columns to avoid NaN correlation
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.loc[:, X_num.std() > 0]
    if X_num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r", "abs_r"])

    corr = X_num.corr(method="pearson")
    cols = corr.columns.tolist()
    records = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            r = corr.loc[a, b]
            if not np.isnan(r) and abs(r) >= threshold:
                records.append({"feature_a": a, "feature_b": b,
                                 "pearson_r": r, "abs_r": abs(r)})
    try:
        df = pd.DataFrame(records).sort_values("abs_r", ascending=False).reset_index(drop=True)
    except:
        df = pd.DataFrame()
    return df


def plot_correlation_heatmap(
    X: pd.DataFrame,
    dataset_name: str,
    event: pd.Series,
    out_path: Path,
    max_features: int = MAX_HEATMAP_FEATURES,
) -> None:
    """Save a feature correlation heatmap, annotating event-correlated features."""
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.loc[:, X_num.std() > 0]

    # If too many features, keep the top-N most correlated with event
    if X_num.shape[1] > max_features:
        ev = event.astype(float)
        scores = X_num.corrwith(ev).abs().sort_values(ascending=False)
        X_num = X_num[scores.index[:max_features]]

    corr = X_num.corr(method="pearson")
    n = len(corr)
    fig_size = max(8, min(24, n * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=max(4, 8 - n // 15))
    ax.set_yticklabels(corr.index, fontsize=max(4, 8 - n // 15))
    ax.set_title(f"{dataset_name} — feature correlation matrix (N={len(X):,})", fontsize=11)

    # Annotate cells with values only for small matrices
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 0.7 else "white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leakage_bar(
    leakage_df: pd.DataFrame,
    dataset_name: str,
    out_path: Path,
    threshold: float,
    top_n: int = 30,
) -> None:
    """Horizontal bar chart of top-N features by leakage score."""
    df = leakage_df.dropna(subset=["abs_score"]).head(top_n).copy()
    if df.empty:
        return

    colors = ["#d73027" if f else "#4575b4" for f in df["flagged"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.32)))
    bars = ax.barh(df["feature"][::-1], df["abs_score"][::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.set_xlabel("|correlation with event|")
    ax.set_title(f"{dataset_name} — label leakage audit (red = flagged ≥ {threshold})")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def analyse_dataset(
    name: str,
    out_dir: Path,
    leakage_thr: float,
    sep_thr: float,
    collinear_thr: float,
) -> list[dict]:
    """Run all three analyses for one dataset. Returns list of flag records."""
    from survpfn.dataloaders import ALL_DATASETS

    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"{'='*60}")

    try:
        df, dur_col, ev_col = ALL_DATASETS[name]()
    except Exception as exc:
        print(f"  [SKIP] Failed to load: {exc}")
        return []

    n, p = df.shape[0], df.shape[1] - 2
    event = (df[ev_col] > 0).astype(int)
    X = df.drop(columns=[dur_col, ev_col])

    print(f"  N={n:,}  features={p}  event_rate={event.mean():.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    flag_records: list[dict] = []

    # ── 1. Leakage ────────────────────────────────────────────────────────
    print("  → leakage analysis …")
    leak_df = leakage_analysis(X, event, threshold=leakage_thr)
    leak_df.to_csv(out_dir / f"{name}_leakage.csv", index=False)
    plot_leakage_bar(leak_df, name, out_dir / f"{name}_leakage.png", leakage_thr)

    n_flagged_leak = leak_df["flagged"].sum()
    print(f"     flagged (|score| ≥ {leakage_thr}): {n_flagged_leak}")
    if n_flagged_leak:
        print(leak_df[leak_df["flagged"]][["feature", "score_type", "score"]].to_string(index=False))

    for _, row in leak_df[leak_df["flagged"]].iterrows():
        flag_records.append({
            "dataset": name, "feature": row["feature"],
            "flag_type": "leakage",
            "score_type": row["score_type"], "score": row["score"],
            "note": f"|{row['score_type']}|={abs(row['score']):.3f} ≥ {leakage_thr}",
        })

    # ── 2. Separation ─────────────────────────────────────────────────────
    print("  → separation analysis …")
    sep_df = separation_analysis(X, event, var_threshold=sep_thr)
    sep_df.to_csv(out_dir / f"{name}_separation.csv", index=False)

    n_flagged_sep = sep_df["flagged"].sum()
    print(f"     flagged (stratum var < {sep_thr}): {n_flagged_sep}")
    if n_flagged_sep:
        print(sep_df[sep_df["flagged"]][["feature", "var_event1", "var_event0", "note"]].to_string(index=False))

    for _, row in sep_df[sep_df["flagged"]].iterrows():
        flag_records.append({
            "dataset": name, "feature": row["feature"],
            "flag_type": "separation",
            "score_type": "min_stratum_var", "score": row["min_var"],
            "note": row["note"],
        })

    # ── 3. Collinearity ───────────────────────────────────────────────────
    print("  → collinearity analysis …")
    coll_df = collinearity_analysis(X, threshold=collinear_thr)
    coll_df.to_csv(out_dir / f"{name}_collinear_pairs.csv", index=False)

    print(f"     flagged pairs (|r| ≥ {collinear_thr}): {len(coll_df)}")
    if not coll_df.empty:
        print(coll_df.head(10).to_string(index=False))

    for _, row in coll_df.iterrows():
        flag_records.append({
            "dataset": name,
            "feature": f"{row['feature_a']} ↔ {row['feature_b']}",
            "flag_type": "collinearity",
            "score_type": "pearson_r", "score": row["pearson_r"],
            "note": f"|r|={row['abs_r']:.3f} ≥ {collinear_thr}",
        })

    # ── 4. Correlation heatmap ────────────────────────────────────────────
    print("  → correlation heatmap …")
    plot_correlation_heatmap(X, name, event, out_dir / f"{name}_corr_heatmap.png")
    print(f"  → saved to {out_dir}/")

    return flag_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset correlation / leakage / separation audit for SurvPFN."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--datasets", nargs="*", default=None,
        help="Explicit dataset name(s) to analyse.",
    )
    grp.add_argument(
        "--group", default=None,
        help="Predefined dataset group (public, ehr, public+ehr, survset, ormoni, all, cr).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SAVE_DIR,
        help=f"Root directory for outputs (default: {SAVE_DIR}).",
    )
    parser.add_argument(
        "--leakage-threshold", type=float, default=LEAKAGE_THRESHOLD,
        help=f"Flag features with |corr(feature, event)| ≥ this value (default {LEAKAGE_THRESHOLD}).",
    )
    parser.add_argument(
        "--separation-var-threshold", type=float, default=SEPARATION_VAR_THR,
        help=(
            f"Flag features whose within-stratum variance < this value (default {SEPARATION_VAR_THR}). "
            "Reproduces the lifelines ConvergenceWarning check."
        ),
    )
    parser.add_argument(
        "--collinear-threshold", type=float, default=COLLINEAR_THRESHOLD,
        help=f"Flag feature pairs with |r| ≥ this value (default {COLLINEAR_THRESHOLD}).",
    )
    parser.add_argument(
        "--skip-survset", action="store_true",
        help="Skip SS_* SurvSet datasets (useful for quick runs).",
    )
    args = parser.parse_args()

    from survpfn.dataloaders import ALL_DATASETS, DATASET_GROUPS

    if args.group:
        if args.group not in DATASET_GROUPS:
            parser.error(f"Unknown group '{args.group}'. "
                         f"Valid: {list(DATASET_GROUPS)}")
        names = DATASET_GROUPS[args.group]
        # Route CR datasets through ALL_DATASETS (which includes CR_DATASETS)
    elif args.datasets:
        names = args.datasets
    else:
        names = list(ALL_DATASETS.keys())
        if args.skip_survset:
            names = [n for n in names if not n.startswith("SS_")]

    print(f"\nCorrelation audit — {len(names)} dataset(s)")
    print(f"  leakage threshold:    {args.leakage_threshold}")
    print(f"  separation threshold: {args.separation_var_threshold}")
    print(f"  collinear threshold:  {args.collinear_threshold}")
    print(f"  output dir:           {args.output_dir}\n")

    all_flags: list[dict] = []
    for name in names:
        flags = analyse_dataset(
            name=name,
            out_dir=args.output_dir / name,
            leakage_thr=args.leakage_threshold,
            sep_thr=args.separation_var_threshold,
            collinear_thr=args.collinear_threshold,
        )
        all_flags.extend(flags)

    # ── Summary CSV ───────────────────────────────────────────────────────
    if all_flags:
        summary_df = pd.DataFrame(all_flags)
        summary_path = args.output_dir / "summary_flags.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"  SUMMARY — {len(summary_df)} flags across {summary_df['dataset'].nunique()} dataset(s)")
        print(f"{'='*60}")
        counts = summary_df.groupby(["dataset", "flag_type"]).size().unstack(fill_value=0)
        print(counts.to_string())
        print(f"\nFull flag table saved to: {summary_path}")

        # Print actionable recommendations
        _print_recommendations(summary_df)
    else:
        print("\nNo flags raised — all datasets look clean.")


def _print_recommendations(df: pd.DataFrame) -> None:
    print("\n── Recommendations ──────────────────────────────────────────")

    leakage = df[df["flag_type"] == "leakage"]
    if not leakage.empty:
        print("\n[LEAKAGE] The following features are strongly correlated with the")
        print("  event indicator and should be reviewed before training:\n")
        for _, row in leakage.iterrows():
            print(f"  • {row['dataset']}/{row['feature']}  ({row['note']})")
        print("\n  → Consider: drop the feature, check for data-entry errors,")
        print("    or verify it is not derived from the outcome.")

    sep = df[df["flag_type"] == "separation"]
    if not sep.empty:
        print("\n[SEPARATION] The following features have near-zero variance in one")
        print("  event stratum (triggers lifelines ConvergenceWarning):\n")
        for _, row in sep.iterrows():
            print(f"  • {row['dataset']}/{row['feature']}  ({row['note']})")
        print("\n  → Consider: drop or binarise the feature, or use a penalised Cox")
        print("    (ridge/Firth) that handles complete separation robustly.")

    coll = df[df["flag_type"] == "collinearity"]
    if not coll.empty:
        print("\n[COLLINEARITY] The following feature pairs are near-redundant (|r| ≥ threshold):\n")
        for _, row in coll.iterrows():
            print(f"  • {row['dataset']}: {row['feature']}  ({row['note']})")
        print("\n  → Consider: drop one of each pair, apply PCA, or use VIF-based pruning.")

    print()


if __name__ == "__main__":
    main()
