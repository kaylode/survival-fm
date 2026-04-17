"""
scripts/significance.py — Statistical significance tests for SurvPFN benchmark results.

Reads ``aggregated.csv`` produced by ``scripts/aggregate.py`` and runs:

1. **Pairwise Wilcoxon signed-rank tests** (fold-level) for every
   (Dataset, Model_A vs Model_B) combination.

2. **Reference-model comparison table** — for each model, compare against
   a configurable set of baselines (default: rsf, cox) and mark * / ** / ***
   significance.

3. **Friedman test** across all datasets jointly — tests whether the overall
   ranking of models differs significantly.

4. **Nemenyi post-hoc test** (via scikit-posthocs if available, otherwise
   Bonferroni-corrected Wilcoxon) — pairwise comparisons pooled over datasets.

5. **Critical-difference (CD) diagram** (saved as PDF if matplotlib available).

Outputs (all written to ``<output_dir>/``)
-----------------------------------------
``significance_pairwise.csv``  — all pairwise (Dataset, A, B, p, significant)
``significance_vs_baselines.csv`` — each model vs each reference
``significance_friedman.csv``  — per-dataset Friedman + omnibus p-value
``significance_nemenyi.csv``   — Nemenyi / Bonferroni-Wilcoxon p-matrix
``significance_summary.txt``   — human-readable summary
``cd_diagram.pdf``             — critical-difference diagram (optional)

CLI
---
    uv run python -m survpfn.scripts.significance
    uv run python -m survpfn.scripts.significance \\
        --aggregated results/benchmark/aggregated.csv \\
        --output-dir results/benchmark \\
        --metric C-index \\
        --references rsf cox deephit_single \\
        --alpha 0.05
"""

from __future__ import annotations

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# ---------------------------------------------------------------------------
# Core test functions
# ---------------------------------------------------------------------------

def _wilcoxon_pair(
    a: np.ndarray,
    b: np.ndarray,
    zero_method: str = "zsplit",
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test on fold values a vs b.

    Returns (statistic, p_value).  With only 5 folds the minimum achievable
    p-value is 0.0625 (all differences in the same direction).
    """
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b, zero_method=zero_method)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def pairwise_wilcoxon(
    df: pd.DataFrame,
    metric: str = "C-index",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All pairwise Wilcoxon tests within each dataset.

    Parameters
    ----------
    df     : aggregated fold-level DataFrame (Dataset, Model, Fold, metrics…)
    metric : metric column to compare
    alpha  : significance threshold

    Returns
    -------
    DataFrame with columns:
        Dataset, Model_A, Model_B, n_folds,
        mean_A, mean_B, mean_diff,
        statistic, p_value, significant, direction
    """
    records = []
    for ds, grp in df.groupby("Dataset"):
        models = grp["Model"].unique()
        for m_a, m_b in combinations(sorted(models), 2):
            vals_a = (
                grp[grp["Model"] == m_a]
                .sort_values("Fold")[metric]
                .dropna()
                .values
            )
            vals_b = (
                grp[grp["Model"] == m_b]
                .sort_values("Fold")[metric]
                .dropna()
                .values
            )
            n = min(len(vals_a), len(vals_b))
            if n < 2:
                continue
            vals_a, vals_b = vals_a[:n], vals_b[:n]

            stat, p = _wilcoxon_pair(vals_a, vals_b)
            diff = float(np.mean(vals_a) - np.mean(vals_b))
            direction = "A>B" if diff > 0 else ("B>A" if diff < 0 else "tie")

            records.append(
                {
                    "Dataset":    ds,
                    "Model_A":    m_a,
                    "Model_B":    m_b,
                    "n_folds":    n,
                    "mean_A":     float(np.mean(vals_a)),
                    "mean_B":     float(np.mean(vals_b)),
                    "mean_diff":  diff,
                    "statistic":  stat,
                    "p_value":    p,
                    "significant": bool(p <= alpha) if not np.isnan(p) else False,
                    "direction":  direction,
                }
            )
    return pd.DataFrame(records)


def vs_references(
    df: pd.DataFrame,
    references: list[str],
    metric: str = "C-index",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """For each (Dataset, model), compare against each reference model.

    Returns a DataFrame with p-values and significance markers.
    Marker convention:
        ***  p ≤ 0.01
        **   p ≤ 0.05
        *    p ≤ 0.0625   (minimum achievable with 5 folds, all concordant)
        (ns) p > 0.0625
    """
    records = []
    for ds, grp in df.groupby("Dataset"):
        models = grp["Model"].unique()
        for ref in references:
            if ref not in models:
                continue
            ref_vals = (
                grp[grp["Model"] == ref].sort_values("Fold")[metric].dropna().values
            )
            for model in sorted(models):
                if model == ref:
                    continue
                mod_vals = (
                    grp[grp["Model"] == model].sort_values("Fold")[metric].dropna().values
                )
                n = min(len(ref_vals), len(mod_vals))
                if n < 2:
                    continue
                rv, mv = ref_vals[:n], mod_vals[:n]
                stat, p = _wilcoxon_pair(mv, rv)
                diff = float(np.mean(mv) - np.mean(rv))

                if np.isnan(p):
                    marker = "?"
                elif p <= 0.01:
                    marker = "***"
                elif p <= 0.05:
                    marker = "**"
                elif p <= 0.0625:
                    marker = "*"
                else:
                    marker = "ns"

                records.append(
                    {
                        "Dataset":     ds,
                        "Reference":   ref,
                        "Model":       model,
                        "n_folds":     n,
                        "mean_ref":    float(np.mean(rv)),
                        "mean_model":  float(np.mean(mv)),
                        "mean_diff":   diff,
                        "p_value":     p,
                        "marker":      marker,
                        "significant": bool(p <= alpha) if not np.isnan(p) else False,
                    }
                )
    return pd.DataFrame(records)


def friedman_test(
    df: pd.DataFrame,
    metric: str = "C-index",
) -> pd.DataFrame:
    """Friedman test across all (Dataset × Model) combinations.

    Treats each fold as a repeated measurement and ranks models within each
    dataset.  Returns per-dataset average ranks plus the omnibus Friedman
    statistic and p-value (pooled across datasets).
    """
    # Build pivot: index=Dataset/Fold, columns=Model
    pivot = (
        df.pivot_table(index=["Dataset", "Fold"], columns="Model", values=metric)
        .dropna(how="all")
    )

    # Per-dataset average ranks (lower rank = better C-index)
    rank_records = []
    groups_for_friedman = {}

    for ds, grp in pivot.groupby(level="Dataset"):
        sub = grp.droplevel("Dataset").dropna(axis=1, how="all")
        if sub.shape[0] < 2 or sub.shape[1] < 2:
            continue
        # Rank within each fold (1 = highest C-index)
        ranked = sub.rank(axis=1, ascending=False, method="average")
        avg_rank = ranked.mean()
        for model, r in avg_rank.items():
            rank_records.append({"Dataset": ds, "Model": model, "avg_rank": r})
        # Store fold values for omnibus test
        for model in sub.columns:
            groups_for_friedman.setdefault(model, []).extend(sub[model].dropna().tolist())

    rank_df = pd.DataFrame(rank_records)

    # Overall average rank across all datasets
    overall_ranks = (
        rank_df.groupby("Model")["avg_rank"]
        .mean()
        .sort_values()
        .reset_index()
        .rename(columns={"avg_rank": "overall_avg_rank"})
    )

    # Omnibus Friedman (fold-averaged values per dataset as observations)
    try:
        fold_mean_pivot = (
            df.groupby(["Dataset", "Model"])[metric]
            .mean()
            .unstack("Model")
            .dropna(how="all")
        )
        fold_mean_pivot = fold_mean_pivot.dropna(axis=1, how="any")
        if fold_mean_pivot.shape[1] >= 2 and fold_mean_pivot.shape[0] >= 2:
            stat, p = friedmanchisquare(
                *[fold_mean_pivot[m].values for m in fold_mean_pivot.columns]
            )
            overall_ranks["friedman_stat"] = stat
            overall_ranks["friedman_p"]    = p
    except Exception as e:
        warnings.warn(f"Friedman test failed: {e}", stacklevel=2)

    return overall_ranks, rank_df


def nemenyi_or_bonferroni(
    df: pd.DataFrame,
    metric: str = "C-index",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Nemenyi post-hoc test (via scikit-posthocs) or Bonferroni-Wilcoxon fallback.

    Averages folds within each dataset, then runs pairwise tests pooled over
    datasets (N = number of datasets for each model pair).

    Returns a symmetric p-value matrix (models × models).
    """
    # Try scikit-posthocs first
    try:
        import scikit_posthocs as sp   # type: ignore
        fold_mean_pivot = (
            df.groupby(["Dataset", "Model"])[metric]
            .mean()
            .unstack("Model")
            .dropna(how="all", axis=0)
        )
        # Nemenyi requires a complete block design; drop models with any NaN
        fold_mean_pivot = fold_mean_pivot.dropna(axis=1, how="any")
        if fold_mean_pivot.shape[1] < 2:
            raise ValueError("Not enough models for Nemenyi after dropping NaNs")
        pmat = sp.posthoc_nemenyi_friedman(fold_mean_pivot.values)
        pmat.index   = fold_mean_pivot.columns
        pmat.columns = fold_mean_pivot.columns
        return pmat.round(4)
    except ImportError:
        pass   # fall through to Bonferroni-Wilcoxon
    except Exception as e:
        warnings.warn(f"Nemenyi failed: {e}; falling back to Bonferroni-Wilcoxon", stacklevel=2)

    # Fallback: pairwise Wilcoxon across datasets, Bonferroni correction
    fold_mean = df.groupby(["Dataset", "Model"])[metric].mean().unstack("Model")
    models = sorted(fold_mean.columns)
    pmat = pd.DataFrame(np.zeros((len(models), len(models))), index=models, columns=models)

    pairs = list(combinations(models, 2))
    raw_p: dict[tuple, float] = {}
    for m_a, m_b in pairs:
        vals = fold_mean[[m_a, m_b]].dropna()
        if len(vals) < 2:
            raw_p[(m_a, m_b)] = float("nan")
            continue
        _, p = _wilcoxon_pair(vals[m_a].values, vals[m_b].values)
        raw_p[(m_a, m_b)] = p

    # Bonferroni correction
    n_tests = len(pairs)
    for (m_a, m_b), p in raw_p.items():
        p_corr = float(min(p * n_tests, 1.0)) if not np.isnan(p) else float("nan")
        pmat.loc[m_a, m_b] = p_corr
        pmat.loc[m_b, m_a] = p_corr

    np.fill_diagonal(pmat.values, 0.0)
    return pmat.round(4)


# ---------------------------------------------------------------------------
# Critical-difference diagram
# ---------------------------------------------------------------------------

def plot_cd_diagram(
    ranks_df: pd.DataFrame,
    friedman_p: float,
    n_datasets: int,
    alpha: float = 0.05,
    output_path: str = "cd_diagram.pdf",
    top_n: int = 20,
) -> None:
    """Draw a Demšar-style critical-difference diagram.

    Parameters
    ----------
    ranks_df    : DataFrame with columns Model, overall_avg_rank.
    friedman_p  : Omnibus Friedman p-value (printed as subtitle).
    n_datasets  : Number of datasets used (for CD computation).
    top_n       : Limit to top_n models by average rank.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        warnings.warn("matplotlib not available; skipping CD diagram.", stacklevel=2)
        return

    # Critical difference (Nemenyi, α=0.05)
    # CD = q_α * sqrt(k(k+1) / (6n))
    # q_alpha values (two-tailed): k=2..20
    # Using approximate critical values
    q_alpha = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
    q = q_alpha.get(alpha, 1.960)

    sub = ranks_df.nsmallest(top_n, "overall_avg_rank").copy()
    k = len(sub)
    if k < 2:
        return
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    sub = sub.sort_values("overall_avg_rank")
    names = sub["Model"].tolist()
    avg_ranks = sub["overall_avg_rank"].tolist()

    fig_height = max(4, k * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1, k + 1)
    ax.set_axis_off()

    # Title
    ax.set_title(
        f"Critical Difference Diagram  (Friedman p={friedman_p:.4f},  "
        f"CD={cd:.2f},  α={alpha})",
        fontsize=10, pad=12,
    )

    # Axis bar
    bar_y = k
    ax.plot([1, k], [bar_y, bar_y], color="black", lw=1.5)
    for r in range(1, k + 1):
        ax.plot([r, r], [bar_y - 0.1, bar_y + 0.1], color="black", lw=1)
        ax.text(r, bar_y + 0.25, str(r), ha="center", va="bottom", fontsize=8)

    # Model names and connectors
    spacing = 0.7
    for i, (name, rank) in enumerate(zip(names, avg_ranks)):
        y_pos = (k - 1 - i) * spacing / max(1, k - 1) * (k - 1)
        ax.plot([rank, rank], [bar_y, y_pos + 0.05], color="gray", lw=0.8, ls="--")
        ax.text(rank, y_pos, f"{name}  ({rank:.2f})", ha="left" if rank <= k / 2 else "right",
                va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))

    # CD bar
    cd_y = -0.5
    mid = np.mean(avg_ranks)
    ax.annotate(
        "", xy=(mid + cd / 2, cd_y), xytext=(mid - cd / 2, cd_y),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
    )
    ax.text(mid, cd_y - 0.3, f"CD = {cd:.2f}", ha="center", va="top",
            fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  CD diagram saved to: {output_path}")


# ---------------------------------------------------------------------------
# Paper-ready LaTeX helpers
# ---------------------------------------------------------------------------

def to_latex_significance_table(
    vs_ref_df: pd.DataFrame,
    reference: str,
    metric: str = "C-index",
    datasets: list[str] | None = None,
) -> str:
    """Return a LaTeX table: rows=models, cols=datasets, cell=mean±marker.

    marker: * p≤0.0625 vs reference, ** p≤0.05, *** p≤0.01
    Bold = best per column.
    """
    sub = vs_ref_df[vs_ref_df["Reference"] == reference].copy()
    if datasets:
        sub = sub[sub["Dataset"].isin(datasets)]

    all_ds = sorted(sub["Dataset"].unique()) if datasets is None else datasets
    models  = sorted(sub["Model"].unique())

    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(all_ds) + r"}")
    lines.append(r"\toprule")
    lines.append("Model & " + " & ".join(all_ds) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        cells = []
        for ds in all_ds:
            row = sub[(sub["Model"] == model) & (sub["Dataset"] == ds)]
            if row.empty:
                cells.append("—")
            else:
                row = row.iloc[0]
                val = f"{row['mean_model']:.3f}"
                cells.append(f"{val}{row['marker'].replace('ns', '')}")
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def make_summary(
    pairwise_df: pd.DataFrame,
    vs_ref_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    alpha: float,
) -> str:
    """Return a human-readable summary of all test results."""
    lines = ["=" * 70]
    lines.append("  SurvPFN Significance Test Summary")
    lines.append("=" * 70)
    lines.append("")

    # Friedman
    if "friedman_p" in friedman_df.columns:
        fp = friedman_df["friedman_p"].iloc[0]
        fs = friedman_df["friedman_stat"].iloc[0]
        lines.append(
            f"Friedman omnibus test: χ²={fs:.3f}, p={fp:.4f} "
            f"({'SIGNIFICANT' if fp <= alpha else 'not significant'} at α={alpha})"
        )
        lines.append("")

    # Top-ranked models by average rank
    lines.append("Average ranks across datasets (lower = better C-index):")
    top = friedman_df.nsmallest(10, "overall_avg_rank")[["Model", "overall_avg_rank"]]
    for _, r in top.iterrows():
        lines.append(f"  {r['Model']:<40s}  avg rank = {r['overall_avg_rank']:.2f}")
    lines.append("")

    # Per-reference significance counts
    if not vs_ref_df.empty:
        lines.append("Models significantly outperforming each baseline (* p≤0.0625):")
        for ref, rgrp in vs_ref_df.groupby("Reference"):
            sig = rgrp[rgrp["marker"].isin(["*", "**", "***"])]
            lines.append(f"  vs {ref}:")
            counts = (
                sig.groupby("Model")["significant"]
                .sum()
                .sort_values(ascending=False)
            )
            for model, cnt in counts.items():
                n_ds = rgrp[rgrp["Model"] == model]["Dataset"].nunique()
                lines.append(f"    {model:<35s}  {cnt}/{n_ds} datasets")
        lines.append("")

    # Pairwise summary
    n_sig = pairwise_df["significant"].sum()
    n_total = len(pairwise_df)
    lines.append(
        f"Pairwise tests: {n_sig}/{n_total} pairs significant at α={alpha} "
        f"({100 * n_sig / max(1, n_total):.1f}%)"
    )
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical significance tests on SurvPFN aggregated results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aggregated", default="results/benchmark/aggregated.csv",
        help="Path to aggregated.csv produced by scripts/aggregate.py.",
    )
    parser.add_argument(
        "--output-dir", default="results/benchmark",
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--metric", default="C-index",
        help="Metric column to use for all tests.",
    )
    parser.add_argument(
        "--references", nargs="+",
        default=["rsf", "cox", "deephit_single"],
        help="Reference models to compare against.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=(
            "Filter to these datasets only. "
            "Keywords: public, ehr, survset, ormoni. "
            "Default: all datasets in the CSV."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Filter to these model keys only. Default: all models in CSV.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold.",
    )
    parser.add_argument(
        "--top-n-cd", type=int, default=20,
        help="Max number of models to show in CD diagram.",
    )
    parser.add_argument(
        "--no-cd", action="store_true",
        help="Skip the critical-difference diagram.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print(f"Loading {args.aggregated} …")
    df = pd.read_csv(args.aggregated)
    print(f"  {len(df)} fold-level rows | "
          f"{df['Dataset'].nunique()} datasets | "
          f"{df['Model'].nunique()} models")

    if args.metric not in df.columns:
        raise ValueError(
            f"Metric '{args.metric}' not found in columns: {list(df.columns)}"
        )

    # Drop rows missing the target metric
    df = df.dropna(subset=[args.metric])

    # ── Dataset keyword expansion ────────────────────────────────────────────
    _DS_KEYWORDS = {
        "public":  ["SUPPORT2", "METABRIC", "GBSG", "WHAS500", "VETERANS", "FLCHAIN", "SEER"],
        "ehr":     ["EICU_SURV", "MIMIC_SURV_B"],
        "survset": [d for d in df["Dataset"].unique() if str(d).startswith("SS_")],
        "ormoni":  ["ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI",
                    "ORMONI_TIRODEI_STROKE", "ORMONI_TIRODEI_MORTALITY"],
    }
    if args.datasets:
        expanded: list[str] = []
        for tok in args.datasets:
            expanded.extend(_DS_KEYWORDS.get(tok, [tok]))
        df = df[df["Dataset"].isin(expanded)]
        print(f"  Filtered to {df['Dataset'].nunique()} datasets: {expanded}")

    if args.models:
        df = df[df["Model"].isin(args.models)]
        print(f"  Filtered to {df['Model'].nunique()} models.")

    # ── 1. Pairwise Wilcoxon ──────────────────────────────────────────
    print("\n[1/4] Pairwise Wilcoxon signed-rank tests …")
    pairwise_df = pairwise_wilcoxon(df, metric=args.metric, alpha=args.alpha)
    p_path = out_dir / "significance_pairwise.csv"
    pairwise_df.to_csv(p_path, index=False)
    print(f"  {len(pairwise_df)} pairs tested | "
          f"{pairwise_df['significant'].sum()} significant at α={args.alpha}")
    print(f"  Saved: {p_path}")

    # ── 2. vs Reference models ────────────────────────────────────────
    refs_present = [r for r in args.references if r in df["Model"].unique()]
    if refs_present:
        print(f"\n[2/4] Comparison vs reference models {refs_present} …")
        vs_ref_df = vs_references(
            df, references=refs_present, metric=args.metric, alpha=args.alpha
        )
        vr_path = out_dir / "significance_vs_baselines.csv"
        vs_ref_df.to_csv(vr_path, index=False)
        print(f"  {len(vs_ref_df)} comparisons | "
              f"markers: " +
              " | ".join(
                  f"{m}={c}" for m, c
                  in vs_ref_df["marker"].value_counts().items()
              ))
        print(f"  Saved: {vr_path}")
    else:
        print(f"\n[2/4] None of {args.references} found in results; skipping.")
        vs_ref_df = pd.DataFrame()

    # ── 3. Friedman test ──────────────────────────────────────────────
    print("\n[3/4] Friedman test …")
    friedman_df, rank_detail_df = friedman_test(df, metric=args.metric)
    fn_path = out_dir / "significance_friedman.csv"
    friedman_df.to_csv(fn_path, index=False)
    rank_detail_df.to_csv(out_dir / "significance_ranks.csv", index=False)
    if "friedman_p" in friedman_df.columns:
        fp = friedman_df["friedman_p"].iloc[0]
        print(f"  Friedman p = {fp:.4f} "
              f"({'significant' if fp <= args.alpha else 'not significant'})")
    print(f"  Saved: {fn_path}")

    # ── 4. Nemenyi / Bonferroni-Wilcoxon ─────────────────────────────
    print("\n[4/4] Nemenyi / Bonferroni-Wilcoxon post-hoc …")
    pmat = nemenyi_or_bonferroni(df, metric=args.metric, alpha=args.alpha)
    nm_path = out_dir / "significance_nemenyi.csv"
    pmat.to_csv(nm_path)
    print(f"  {pmat.shape[0]}×{pmat.shape[1]} p-value matrix saved: {nm_path}")

    # ── Summary text ─────────────────────────────────────────────────
    summary = make_summary(pairwise_df, vs_ref_df, friedman_df, args.alpha)
    print("\n" + summary)
    summary_path = out_dir / "significance_summary.txt"
    summary_path.write_text(summary)
    print(f"\n  Summary saved: {summary_path}")

    # ── CD diagram ────────────────────────────────────────────────────
    if not args.no_cd:
        fp_val = (
            float(friedman_df["friedman_p"].iloc[0])
            if "friedman_p" in friedman_df.columns
            else float("nan")
        )
        n_ds = df["Dataset"].nunique()
        cd_path = str(out_dir / "cd_diagram.pdf")
        plot_cd_diagram(
            friedman_df, fp_val, n_ds,
            alpha=args.alpha,
            output_path=cd_path,
            top_n=args.top_n_cd,
        )


if __name__ == "__main__":
    main()
