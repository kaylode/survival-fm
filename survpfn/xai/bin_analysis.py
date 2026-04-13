"""
Bin-count analysis for discrete-time survival models.

For each dataset:
  1. Histogram of event times with quantile-bin boundaries overlaid
  2. Events-per-bin for candidate bin counts [10, 20, 30, 50, 100]
  3. Valid-samples-per-bin (E=1 OR T > t_k) for candidate bin counts

Run:
    uv run python xai/bin_analysis.py [--datasets SUPPORT2 ORMONI_TIRODEI_MORTALITY ...]
                                      [--bins 10 20 30 50 100]
                                      [--out xai/bin_plots]
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Datasets to analyse
# ---------------------------------------------------------------------------
from survpfn.dataloaders import ALL_DATASETS

DEFAULT_DATASETS = sorted(list(ALL_DATASETS.keys()))
DEFAULT_BINS = [10, 20, 30, 50, 100, 300, 500, 1000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantile_cuts(times: np.ndarray, n_bins: int) -> np.ndarray:
    """Return n_bins+1 quantile-based cut-points (including min/max)."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    cuts = np.unique(np.percentile(times, quantiles))
    return cuts


def compute_events_per_bin(T: np.ndarray, E: np.ndarray, cuts: np.ndarray):
    """
    For each interval (cuts[k], cuts[k+1]], count events.
    Returns array of length len(cuts)-1.
    """
    n_bins = len(cuts) - 1
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        lo, hi = cuts[k], cuts[k + 1]
        mask = (T > lo) & (T <= hi) & (E == 1)
        counts[k] = mask.sum()
    return counts


def compute_valid_samples_per_bin(T: np.ndarray, E: np.ndarray, cuts: np.ndarray):
    """
    valid_samples_k = #{E=1 OR T > cuts[k+1]}
    Patients who either had an event at any point, or are still at risk after bin k.
    This reflects how many patients can provide a meaningful label for bin k.
    """
    n_bins = len(cuts) - 1
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        hi = cuts[k + 1]
        mask = (E == 1) | (T > hi)
        counts[k] = mask.sum()
    return counts


def load_dataset(name: str):
    """Load dataset, returning (T, E, label) where T=times, E=events (binary: >0)."""
    from survpfn.dataloaders import get_dataset
    df, dur_col, ev_col = get_dataset(name)
    T = df[dur_col].values.astype(float)
    E_raw = df[ev_col].values
    # For competing risks (values > 1), treat any event as event=1
    E = (E_raw > 0).astype(int)
    label = f"{name}  (N={len(T)}, events={E.sum()})"
    return T, E, label


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyse_dataset(name: str, bin_counts: list[int], out_dir: Path):
    print(f"  Loading {name} ...", flush=True)
    try:
        T, E, label = load_dataset(name)
    except Exception as exc:
        print(f"    SKIP — {exc}")
        return None

    T_events = T[E == 1]
    if len(T_events) == 0:
        print(f"    SKIP — no events found")
        return None

    n_total = len(T)
    n_events = int(E.sum())
    n_censored = n_total - n_events
    t_min, t_max = T_events.min(), T_events.max()
    t_median = np.median(T_events)

    # -----------------------------------------------------------------------
    # Figure layout: top row = histogram; subsequent rows = events + valid per bin count
    # -----------------------------------------------------------------------
    n_bin_counts = len(bin_counts)
    n_rows = 1 + n_bin_counts   # histogram row + one row per bin count
    fig = plt.figure(figsize=(14, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ---- Row 0: event-time histogram with 10-bin boundaries overlaid ------
    ax_hist = fig.add_subplot(gs[0, :])
    cuts_ref = quantile_cuts(T_events, 10)
    ax_hist.hist(T_events, bins=60, color="#4C72B0", alpha=0.75,
                 edgecolor="white", linewidth=0.4, label="Event times")
    for k, c in enumerate(cuts_ref[1:-1]):
        ax_hist.axvline(c, color="#DD3333", lw=0.9, ls="--", alpha=0.7,
                        label="10-bin boundaries" if k == 0 else None)
    ax_hist.set_xlabel("Time")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"[{name}] Event-time distribution\n{label}", fontsize=11)
    ax_hist.legend(fontsize=8)

    # ---- Rows 1..n: events-per-bin  &  valid-samples-per-bin -------------
    summary_rows = []

    for row_idx, n_bins in enumerate(bin_counts, start=1):
        cuts = quantile_cuts(T_events, n_bins)
        actual_bins = len(cuts) - 1  # may differ if quantiles collapse
        bin_indices = np.arange(actual_bins)

        events_per_bin = compute_events_per_bin(T, E, cuts)
        valid_per_bin = compute_valid_samples_per_bin(T, E, cuts)

        sparse_bins = int((events_per_bin < 5).sum())
        empty_bins = int((events_per_bin == 0).sum())
        min_epb = int(events_per_bin.min())
        median_epb = float(np.median(events_per_bin))

        summary_rows.append({
            "dataset": name,
            "n_bins_requested": n_bins,
            "n_bins_actual": actual_bins,
            "total_events": n_events,
            "min_events_per_bin": min_epb,
            "median_events_per_bin": median_epb,
            "bins_lt5_events": sparse_bins,
            "empty_bins": empty_bins,
            "valid_samples_last_bin": int(valid_per_bin[-1]),
            "valid_samples_drop_pct": round(
                100 * (1 - valid_per_bin[-1] / valid_per_bin[0]), 1
            ) if valid_per_bin[0] > 0 else 0.0,
        })

        # Events per bin
        ax_ev = fig.add_subplot(gs[row_idx, 0])
        colors_ev = ["#DD3333" if v < 5 else "#4C72B0" for v in events_per_bin]
        ax_ev.bar(bin_indices, events_per_bin, color=colors_ev, edgecolor="white",
                  linewidth=0.3, width=0.85)
        ax_ev.axhline(5, color="#DD3333", ls="--", lw=1.0, label="<5 threshold")
        ax_ev.axhline(20, color="orange", ls="--", lw=1.0, label="20 target")
        ax_ev.set_xlabel("Bin index")
        ax_ev.set_ylabel("Events")
        ax_ev.set_title(f"Events/bin  |  n_bins={n_bins}  (sparse={sparse_bins})",
                        fontsize=9)
        ax_ev.legend(fontsize=7)

        # Valid samples per bin
        ax_vs = fig.add_subplot(gs[row_idx, 1])
        ax_vs.plot(bin_indices, valid_per_bin, color="#2CA02C", lw=1.5,
                   marker="o", ms=3, label="valid samples")
        ax_vs.fill_between(bin_indices, valid_per_bin, alpha=0.15, color="#2CA02C")
        ax_vs.set_xlabel("Bin index")
        ax_vs.set_ylabel("Valid samples")
        ax_vs.set_title(
            f"Valid-samples/bin  |  n_bins={n_bins}  "
            f"(drop {summary_rows[-1]['valid_samples_drop_pct']:.0f}%)",
            fontsize=9,
        )

    fig.suptitle(
        f"{name}  |  N={n_total}, events={n_events} ({100*n_events/n_total:.0f}%), "
        f"censored={n_censored}\n"
        f"Event-time range [{t_min:.1f}, {t_max:.1f}], median={t_median:.1f}",
        fontsize=11, y=1.01,
    )

    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")

    return summary_rows


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def plot_summary_heatmap(df: pd.DataFrame, out_dir: Path, metric: str,
                         cmap: str, title: str, fname: str, fmt: str = ".0f"):
    pivot = df.pivot(index="dataset", columns="n_bins_requested", values=metric)
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                    max(4, len(pivot) * 0.55)))
    vmax = pivot.values[np.isfinite(pivot.values)].max() if fmt != ".0f" else None
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Bin count")
    ax.set_title(title, fontsize=11)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text = f"{val:{fmt}}" if np.isfinite(val) else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if val < (pivot.values.max() * 0.6) else "black")
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap → {out_dir / fname}")


# ---------------------------------------------------------------------------
# Recommended bins per dataset
# ---------------------------------------------------------------------------

def recommend_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based recommendation:
      - Find largest n_bins where min_events_per_bin >= 5 AND empty_bins == 0
      - If none satisfy, pick the n_bins with highest min_events_per_bin
    """
    recs = []
    for dataset, grp in df.groupby("dataset"):
        grp = grp.sort_values("n_bins_requested")
        ok = grp[(grp["min_events_per_bin"] >= 5) & (grp["empty_bins"] == 0)]
        if len(ok) > 0:
            rec = int(ok["n_bins_requested"].max())
            reason = f"largest n_bins with min_EPB≥5 & no empty bins"
        else:
            best = grp.loc[grp["min_events_per_bin"].idxmax()]
            rec = int(best["n_bins_requested"])
            reason = f"best available (min_EPB={best['min_events_per_bin']:.0f})"
        recs.append({
            "dataset": dataset,
            "recommended_bins": rec,
            "reason": reason,
            "total_events": int(grp["total_events"].iloc[0]),
        })
    return pd.DataFrame(recs).sort_values("dataset")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bin-count analysis for survival models")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Dataset names (default: public + ORMONI_TIRODEI)")
    parser.add_argument("--bins", nargs="+", type=int, default=DEFAULT_BINS,
                        help="Bin counts to test (default: 10 20 30 50 100)")
    parser.add_argument("--out", default="results/xai/datasets/bin_plots",
                        help="Output directory (default: xai/bin_plots)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    failed = []

    for name in args.datasets:
        print(f"\n[{name}]")
        rows = analyse_dataset(name, sorted(args.bins), out_dir)
        if rows:
            all_summary.extend(rows)
        else:
            failed.append(name)

    if not all_summary:
        print("\nNo datasets analysed — check dataset availability.")
        return

    df_sum = pd.DataFrame(all_summary)

    # Save raw summary
    csv_path = out_dir / "bin_analysis_summary.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"\nSummary CSV → {csv_path}")

    # Heatmaps
    print("\nGenerating summary heatmaps ...")
    plot_summary_heatmap(
        df_sum, out_dir,
        metric="min_events_per_bin", cmap="RdYlGn",
        title="Min events per bin  (green = ≥20, red = <5)",
        fname="heatmap_min_events_per_bin.png",
    )
    plot_summary_heatmap(
        df_sum, out_dir,
        metric="bins_lt5_events", cmap="RdYlGn_r",
        title="# bins with < 5 events  (green = 0 sparse bins)",
        fname="heatmap_sparse_bins.png",
    )
    plot_summary_heatmap(
        df_sum, out_dir,
        metric="valid_samples_drop_pct", cmap="RdYlGn_r",
        title="Valid-samples drop % from bin-0 to last bin",
        fname="heatmap_valid_drop.png",
        fmt=".1f",
    )

    # Recommendations
    df_rec = recommend_bins(df_sum)
    rec_path = out_dir / "recommendations.csv"
    df_rec.to_csv(rec_path, index=False)

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BIN-COUNT ANALYSIS REPORT")
    print("=" * 72)

    for _, row in df_rec.iterrows():
        ds = row["dataset"]
        grp = df_sum[df_sum["dataset"] == ds].sort_values("n_bins_requested")
        print(f"\n  {ds}  (events={row['total_events']})")
        print(f"    {'bins':>6}  {'min_EPB':>8}  {'median_EPB':>11}  "
              f"{'sparse(<5)':>10}  {'empty':>6}  {'valid_drop%':>11}")
        for _, r in grp.iterrows():
            flag = " ← REC" if r["n_bins_requested"] == row["recommended_bins"] else ""
            print(f"    {r['n_bins_requested']:>6}  {r['min_events_per_bin']:>8.0f}  "
                  f"{r['median_events_per_bin']:>11.1f}  "
                  f"{r['bins_lt5_events']:>10}  {r['empty_bins']:>6}  "
                  f"{r['valid_samples_drop_pct']:>10.1f}%{flag}")

    print("\n" + "-" * 72)
    print("RECOMMENDATIONS SUMMARY")
    print("-" * 72)
    for _, row in df_rec.iterrows():
        print(f"  {row['dataset']:<25} → {row['recommended_bins']:>4} bins   ({row['reason']})")

    if failed:
        print(f"\n  FAILED to load: {failed}")

    print(f"\nAll outputs saved to: {out_dir.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
