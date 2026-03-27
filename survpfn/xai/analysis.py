"""
survpfn/xai/analysis.py — Comprehensive results analysis for SurvPFN.

Generates a full suite of publication-quality figures directly from the
per-fold result folders produced by scripts/benchmark.py. No aggregated
CSV is required — data is loaded on the fly.

Figures produced
----------------
fig01_heatmap_cindex         Mean C-index heatmap (models × datasets)
fig02_cindex_comparison      C-index bar charts grouped by dataset
fig03_ibs_comparison         IBS bar charts (lower is better)
fig04_multimetric_overview   Box plots: C-index / IBS / AUC / D-cal
fig05_auc_curves             Time-dependent AUROC per dataset
fig06_efficiency_frontier    C-index vs training time (log-scale scatter)
fig07_model_family_boxplot   C-index box plots per model group
fig08_tabpfn_ablation        TabPFN variants vs classical counterparts
fig09_sirbu_multitask        Performance across the 4 Sirbu outcomes
fig10_feature_importance_*   Cox hazard ratios + tree importances per dataset
fig11_hpo_convergence        Optuna best-value vs n_trials
fig11b_hpo_convergence_speed Optuna convergence speed (best trial / n_trials)
fig12_ranking                Average rank heatmap + sorted bar chart
fig13_dcal_heatmap           D-calibration heatmap

Usage
-----
    uv run python -m survpfn.xai.analysis
    uv run python -m survpfn.xai.analysis --results-dir results --output-dir xai/figures
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Palette & ordering constants
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "km", "cox",
    "rsf", "gbsa",
    "deepsurv", "mtlr", "pchazard", "deephit_single",
    "embedding_cox",
    "surv_cox", "surv_deephit", "surv_pchazard", "surv_mtlr",
]

MODEL_LABELS = {
    "km":             "KM",
    "cox":            "Cox PH",
    "rsf":            "RSF",
    "gbsa":           "GBSA",
    "deepsurv":       "DeepSurv",
    "mtlr":           "MTLR",
    "pchazard":       "PC-Hazard",
    "deephit_single": "DeepHit",
    "embedding_cox":  "TabPFN-Cox",
    "surv_cox":       "Surv-Cox",
    "surv_deephit":   "Surv-DeepHit",
    "surv_pchazard":  "Surv-PCHaz",
    "surv_mtlr":      "Surv-MTLR",
}

MODEL_GROUPS = {
    "Baseline": ["km", "cox"],
    "Tree":     ["rsf", "gbsa"],
    "Deep":     ["deepsurv", "mtlr", "pchazard", "deephit_single"],
    "TabPFN-A": ["embedding_cox"],
    "TabPFN-C": ["surv_cox", "surv_deephit", "surv_pchazard", "surv_mtlr"],
}
MODEL_TO_GROUP = {m: g for g, ms in MODEL_GROUPS.items() for m in ms}

GROUP_COLORS = {
    "Baseline": "#4e79a7",
    "Tree":     "#f28e2b",
    "Deep":     "#59a14f",
    "TabPFN-A": "#e15759",
    "TabPFN-C": "#9467bd",
}

DATASET_ORDER = [
    "SUPPORT2", "METABRIC", "GBSG",
    "SIRBU_mortality", "SIRBU_cv", "SIRBU_mi", "SIRBU_stroke",
    "SIRBU",
]
DATASET_LABELS = {
    "GBSG":           "GBSG",
    "METABRIC":       "METABRIC",
    "SUPPORT2":       "SUPPORT2",
    "SIRBU":          "SIRBU",
    "SIRBU_mortality":"Sirbu-Mort.",
    "SIRBU_cv":       "Sirbu-CV",
    "SIRBU_mi":       "Sirbu-MI",
    "SIRBU_stroke":   "Sirbu-Stroke",
}
PUBLIC_DATASETS = ["SUPPORT2", "METABRIC", "GBSG"]
SIRBU_DATASETS  = ["SIRBU_mortality", "SIRBU_cv", "SIRBU_mi", "SIRBU_stroke", "SIRBU"]

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    """Build a flat DataFrame from all metrics.json + metadata.json files."""
    records = []
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        parts = metrics_path.parts
        try:
            dataset = parts[-4]
            model   = parts[-3]
            fold    = int(parts[-2].replace("fold_", ""))
        except (IndexError, ValueError):
            continue
        if dataset not in DATASET_ORDER:
            continue

        with metrics_path.open() as f:
            metrics = json.load(f)

        meta = {}
        meta_path = metrics_path.parent / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)

        record = {"Dataset": dataset, "Model": model, "Fold": fold}
        record.update(metrics)
        for key in ("fit_time_s", "eval_time_s", "n_params",
                    "n_train", "n_test", "n_features",
                    "n_events_train", "n_events_test",
                    "event_rate_train", "event_rate_test"):
            record[key] = meta.get(key)
        records.append(record)

    df = pd.DataFrame(records)
    df["Group"]   = df["Model"].map(MODEL_TO_GROUP)
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATASET_ORDER, ordered=True)
    present_models = [m for m in MODEL_ORDER if m in df["Model"].unique()]
    df["Model"]   = pd.Categorical(df["Model"], categories=present_models, ordered=True)
    # Convert None → NaN for all metric columns
    for col in df.columns:
        if col not in ("Dataset", "Model", "Fold", "Group"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["Dataset", "Model", "Fold"]).reset_index(drop=True)


def load_feature_importance(results_dir: Path) -> list[dict]:
    records = []
    for fi_path in sorted(results_dir.rglob("feature_importance.json")):
        parts = fi_path.parts
        try:
            dataset = parts[-4]
            model   = parts[-3]
            fold    = int(parts[-2].replace("fold_", ""))
        except (IndexError, ValueError):
            continue
        if dataset not in DATASET_ORDER:
            continue
        with fi_path.open() as f:
            fi = json.load(f)
        records.append({"dataset": dataset, "model": model, "fold": fold, **fi})
    return records


def load_best_params(results_dir: Path) -> pd.DataFrame:
    records = []
    for bp_path in sorted(results_dir.rglob("best_params.json")):
        parts = bp_path.parts
        try:
            dataset = parts[-4]
            model   = parts[-3]
            fold    = int(parts[-2].replace("fold_", ""))
        except (IndexError, ValueError):
            continue
        if dataset not in DATASET_ORDER:
            continue
        with bp_path.open() as f:
            bp = json.load(f)
        records.append({
            "Dataset": dataset, "Model": model, "Fold": fold,
            "best_value":  bp.get("best_value"),
            "n_trials":    bp.get("n_completed_trials"),
            "best_trial":  bp.get("best_trial_number"),
            **{f"param_{k}": v for k, v in bp.get("best_params", {}).items()},
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_color(model: str) -> str:
    return GROUP_COLORS.get(MODEL_TO_GROUP.get(model, ""), "#888888")


def present(df: pd.DataFrame, lst: list[str]) -> list[str]:
    """Filter a list to only items present in df."""
    return [x for x in lst if x in df.values]


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def group_separator_lines(ax, group_sizes, orientation="h"):
    """Draw subtle separator lines between model groups."""
    pos = -0.5
    for gs in group_sizes[:-1]:
        pos += gs
        if orientation == "h":
            ax.axhline(pos, color="white", linewidth=2)
        else:
            ax.axvline(pos, color="gray", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Figure 01 — C-index heatmap
# ---------------------------------------------------------------------------

def fig01_heatmap_cindex(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df["Model"].unique()]
    pivot = (df.groupby(["Model", "Dataset"], observed=True)["C-index"]
               .mean()
               .unstack("Dataset")
               .reindex(index=models, columns=datasets))

    fig, ax = plt.subplots(figsize=(len(datasets) * 1.4 + 1.5, len(models) * 0.65 + 2))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", vmin=0.45, vmax=0.85, aspect="auto")
    plt.colorbar(im, ax=ax, label="C-index (↑ better)", shrink=0.75)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])

    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            val = pivot.loc[m, d]
            if not np.isnan(val):
                txt_col = "white" if val < 0.57 or val > 0.78 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=txt_col)

    # Group dividers (Baseline=2, Tree=2, Deep=4, TabPFN-A=1, TabPFN-C=4)
    group_separator_lines(ax, [2, 2, 4, 1, 4], "h")
    # Add group labels on the left
    group_rows = {"Baseline": 0.5, "Tree": 2.5, "Deep": 5, "TabPFN-A": 8, "TabPFN-C": 10}
    for g, y in group_rows.items():
        if any(m in df["Model"].unique() for m in MODEL_GROUPS.get(g, [])):
            ax.text(-0.8, y, g, ha="right", va="center", fontsize=7.5,
                    color=GROUP_COLORS.get(g, "black"), fontweight="bold")

    ax.set_title("Mean C-index Across 5-fold CV", fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig01_heatmap_cindex")


# ---------------------------------------------------------------------------
# Figure 02 — C-index bar charts per dataset
# ---------------------------------------------------------------------------

def fig02_cindex_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df["Model"].unique()]
    n_cols = min(4, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols

    summary = (df.groupby(["Dataset", "Model"], observed=True)["C-index"]
                 .agg(["mean", "std"]).reset_index())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(models).dropna(subset=["mean"]))
        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        if "cox" in sub.index:
            ax.axhline(sub.loc["cox", "mean"], color="#4e79a7",
                       linestyle="--", linewidth=1, alpha=0.6, label="Cox baseline")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("C-index" if idx % n_cols == 0 else "")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        ymax = min(1.0, sub["mean"].max() + 0.08)
        ax.set_ylim(0.3, ymax)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODEL_GROUPS),
               bbox_to_anchor=(0.5, -0.01), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("C-index Comparison (mean ± std, 5-fold CV)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig02_cindex_comparison")


# ---------------------------------------------------------------------------
# Figure 03 — IBS bar charts
# ---------------------------------------------------------------------------

def fig03_ibs_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    df_ibs = df.dropna(subset=["IBS"])
    if df_ibs.empty:
        print("  [skip] fig03 — no IBS data")
        return
    datasets = [d for d in DATASET_ORDER if d in df_ibs["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df_ibs["Model"].unique()]
    n_cols = min(4, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    summary = (df_ibs.groupby(["Dataset", "Model"], observed=True)["IBS"]
                     .agg(["mean", "std"]).reset_index())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(models).dropna(subset=["mean"]))
        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        if "cox" in sub.index:
            ax.axhline(sub.loc["cox", "mean"], color="#4e79a7",
                       linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("IBS (↓ better)" if idx % n_cols == 0 else "")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODEL_GROUPS),
               bbox_to_anchor=(0.5, -0.01), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("Integrated Brier Score — Lower is Better (mean ± std, 5-fold CV)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig03_ibs_comparison")


# ---------------------------------------------------------------------------
# Figure 04 — Multi-metric box plots (all datasets pooled)
# ---------------------------------------------------------------------------

def fig04_multimetric_overview(df: pd.DataFrame, out_dir: Path) -> None:
    metrics_cfg = [
        ("C-index",  "C-index (↑)",  (0.3, 1.0)),
        ("IBS",      "IBS (↓)",      (0.0, 0.5)),
        ("AUC_mean", "AUC mean (↑)", (0.3, 1.0)),
        ("D-cal",    "D-cal (↑)",    (0.0, 1.05)),
    ]
    models = [m for m in MODEL_ORDER if m in df["Model"].unique()]
    colors = [get_model_color(m) for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (col, label, ylim) in zip(axes, metrics_cfg):
        sub = df.dropna(subset=[col])
        if sub.empty:
            ax.set_visible(False)
            continue
        data = [sub[sub["Model"] == m][col].dropna().values for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(label, fontweight="bold")
        # Group separators
        for sep in [2.5, 4.5, 8.5, 9.5]:
            if sep < len(models):
                ax.axvline(sep, color="gray", linestyle=":", alpha=0.4)
        # Sample sizes
        for i, (m, d) in enumerate(zip(models, data), 1):
            ax.text(i, ylim[0] + (ylim[1]-ylim[0])*0.02, f"n={len(d)}",
                    ha="center", fontsize=6.5, color="gray")

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODEL_GROUPS),
               bbox_to_anchor=(0.5, -0.02), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("Multi-metric Performance Overview (all datasets pooled)",
                 fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_dir, "fig04_multimetric_overview")


# ---------------------------------------------------------------------------
# Figure 05 — Time-dependent AUROC curves
# ---------------------------------------------------------------------------

def fig05_auc_curves(df: pd.DataFrame, out_dir: Path) -> None:
    auc_cols = sorted([c for c in df.columns if c.startswith("AUC_t=")],
                      key=lambda c: float(c.split("=")[1]))
    if not auc_cols:
        print("  [skip] fig05 — no AUC_t= columns")
        return
    datasets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df["Model"].unique()]

    n_cols = min(4, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub_ds = df[df["Dataset"] == dataset]
        t_cols = [c for c in auc_cols if sub_ds[c].notna().any()]
        if not t_cols:
            ax.set_visible(False)
            continue
        times = np.array([float(c.split("=")[1]) for c in t_cols])

        for model in models:
            sub_m = sub_ds[sub_ds["Model"] == model]
            if sub_m.empty:
                continue
            auc_vals = sub_m[t_cols].mean()
            valid    = auc_vals.notna().values
            if valid.sum() < 2:
                continue
            ax.plot(times[valid], auc_vals.values[valid],
                    marker="o", markersize=2.5, linewidth=1.5,
                    label=MODEL_LABELS.get(model, model),
                    color=get_model_color(model))

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("AUC" if idx % n_cols == 0 else "")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3)

    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles, labels = [], []
    for m in models:
        handles.append(plt.Line2D([0], [0], color=get_model_color(m), linewidth=1.5,
                                  marker="o", markersize=3))
        labels.append(MODEL_LABELS.get(m, m))
    fig.legend(handles, labels, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=8)
    fig.suptitle("Time-dependent AUROC (mean across 5 folds)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, out_dir, "fig05_auc_curves")


# ---------------------------------------------------------------------------
# Figure 06 — Efficiency frontier
# ---------------------------------------------------------------------------

def fig06_efficiency_frontier(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df.dropna(subset=["C-index", "fit_time_s"])
    if sub.empty:
        print("  [skip] fig06 — no timing data")
        return
    agg = (sub.groupby(["Model", "Dataset", "Group"], observed=True)
              .agg(cindex=("C-index", "mean"), fit_time=("fit_time_s", "mean"))
              .reset_index())

    fig, ax = plt.subplots(figsize=(11, 6))
    for group, color in GROUP_COLORS.items():
        g = agg[agg["Group"] == group]
        ax.scatter(g["fit_time"], g["cindex"], color=color, s=80,
                   alpha=0.75, label=group, edgecolors="white", linewidth=0.5, zorder=3)

    # Annotate by model (average position across datasets)
    model_agg = agg.groupby("Model", observed=True).agg(
        cindex=("cindex", "mean"), fit_time=("fit_time", "mean")).reset_index()
    for _, row in model_agg.iterrows():
        ax.annotate(MODEL_LABELS.get(row["Model"], row["Model"]),
                    (row["fit_time"], row["cindex"]),
                    fontsize=7.5, xytext=(4, 3), textcoords="offset points", alpha=0.8)

    # Pareto frontier
    srt = agg.sort_values("fit_time")
    px, py, best_c = [], [], -np.inf
    for _, row in srt.iterrows():
        if row["cindex"] > best_c:
            best_c = row["cindex"]
            px.append(row["fit_time"])
            py.append(row["cindex"])
    if len(px) > 1:
        ax.step(px, py, where="post", color="crimson", linewidth=2,
                linestyle="--", label="Pareto frontier", alpha=0.7, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (seconds, log scale, includes HPO when tuned)")
    ax.set_ylabel("Mean C-index")
    ax.set_title("Performance–Efficiency Frontier\n(each point = model × dataset mean)",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig06_efficiency_frontier")


# ---------------------------------------------------------------------------
# Figure 07 — Model family box plots (public vs Sirbu)
# ---------------------------------------------------------------------------

def fig07_model_family_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    groups = list(MODEL_GROUPS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (split_datasets, title) in zip(axes, [
        ([d for d in PUBLIC_DATASETS if d in df["Dataset"].unique()], "Public Datasets"),
        ([d for d in SIRBU_DATASETS  if d in df["Dataset"].unique()], "Sirbu (Private)"),
    ]):
        if not split_datasets:
            ax.set_visible(False)
            continue
        sub = df[df["Dataset"].isin(split_datasets)].copy()
        sub["Group"] = sub["Model"].map(MODEL_TO_GROUP)
        data   = [sub[sub["Group"] == g]["C-index"].dropna().values for g in groups]
        colors = [GROUP_COLORS[g] for g in groups]

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=4, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=15)
        ax.set_ylabel("C-index")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        for i, (g, d) in enumerate(zip(groups, data), 1):
            ax.text(i, 0.31, f"n={len(d)}", ha="center", fontsize=7, color="gray")

    fig.suptitle("C-index Distribution by Model Family (Public vs Private)",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig07_model_family_boxplot")


# ---------------------------------------------------------------------------
# Figure 08 — TabPFN ablation
# ---------------------------------------------------------------------------

def fig08_tabpfn_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("cox",           "embedding_cox",  "Cox PH",   "TabPFN-Cox"),
        ("cox",           "surv_cox",       "Cox PH",   "Surv-Cox"),
        ("mtlr",          "surv_mtlr",      "MTLR",     "Surv-MTLR"),
        ("deephit_single","surv_deephit",   "DeepHit",  "Surv-DeepHit"),
    ]
    valid_pairs = [(a, b, la, lb) for a, b, la, lb in pairs
                   if a in df["Model"].unique() and b in df["Model"].unique()]
    if not valid_pairs:
        print("  [skip] fig08 — no matching model pairs")
        return

    datasets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    fig, axes = plt.subplots(len(valid_pairs), 1,
                             figsize=(13, len(valid_pairs) * 3.5), squeeze=False)

    for ax, (base_m, tabpfn_m, base_l, tabpfn_l) in zip(axes.flatten(), valid_pairs):
        base_agg   = (df[df["Model"] == base_m]
                      .groupby("Dataset", observed=True)["C-index"].agg(["mean","std"]))
        tabpfn_agg = (df[df["Model"] == tabpfn_m]
                      .groupby("Dataset", observed=True)["C-index"].agg(["mean","std"]))
        ds = [d for d in datasets if d in base_agg.index or d in tabpfn_agg.index]
        x = np.arange(len(ds)); w = 0.35

        b_m  = [base_agg.loc[d,"mean"]   if d in base_agg.index   else np.nan for d in ds]
        b_s  = [base_agg.loc[d,"std"]    if d in base_agg.index   else 0       for d in ds]
        t_m  = [tabpfn_agg.loc[d,"mean"] if d in tabpfn_agg.index else np.nan for d in ds]
        t_s  = [tabpfn_agg.loc[d,"std"]  if d in tabpfn_agg.index else 0       for d in ds]

        ax.bar(x - w/2, b_m, w, yerr=b_s, label=base_l,
               color=get_model_color(base_m), capsize=3, edgecolor="white")
        tabpfn_col = (GROUP_COLORS["TabPFN-C"] if tabpfn_m.startswith("surv_")
                      else GROUP_COLORS["TabPFN-A"])
        ax.bar(x + w/2, t_m, w, yerr=t_s, label=tabpfn_l,
               color=tabpfn_col, capsize=3, edgecolor="white")

        for xi, (bv, tv) in enumerate(zip(b_m, t_m)):
            if not (np.isnan(bv) or np.isnan(tv)):
                delta = tv - bv
                col   = "#2ca02c" if delta > 0.01 else ("#d62728" if delta < -0.01 else "gray")
                ypos  = max(bv, tv) + max(b_s[xi] if b_s else 0, t_s[xi] if t_s else 0) + 0.012
                ax.text(xi, ypos, f"{delta:+.3f}", ha="center", fontsize=7.5,
                        color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in ds], rotation=30, ha="right")
        ax.set_ylabel("C-index")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"{base_l}  →  {tabpfn_l}", fontweight="bold")

    fig.suptitle("TabPFN Ablation: Classical vs Foundation Model Counterparts\n"
                 "(Δ C-index shown above bars; green = improvement, red = degradation)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig08_tabpfn_ablation")


# ---------------------------------------------------------------------------
# Figure 09 — Sirbu multi-task
# ---------------------------------------------------------------------------

def fig09_sirbu_multitask(df: pd.DataFrame, out_dir: Path) -> None:
    sirbu_ds = [d for d in SIRBU_DATASETS if d in df["Dataset"].unique() and d != "SIRBU"]
    if not sirbu_ds:
        print("  [skip] fig09 — no Sirbu multi-task data")
        return
    models = [m for m in MODEL_ORDER
              if m in df[df["Dataset"].isin(sirbu_ds)]["Model"].unique()
              and m != "km"]

    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(sirbu_ds)
    x = np.arange(n)
    w = 0.7 / len(models)
    offsets = np.linspace(-0.35 + w/2, 0.35 - w/2, len(models))

    for model, offset in zip(models, offsets):
        sub = df[df["Dataset"].isin(sirbu_ds) & (df["Model"] == model)]
        means = sub.groupby("Dataset", observed=True)["C-index"].mean().reindex(sirbu_ds)
        stds  = sub.groupby("Dataset", observed=True)["C-index"].std().reindex(sirbu_ds)
        ax.bar(x + offset, means, w * 0.92, yerr=stds,
               label=MODEL_LABELS.get(model, model),
               color=get_model_color(model), capsize=2,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in sirbu_ds])
    ax.set_ylabel("C-index")
    ax.set_ylim(0.3, 0.92)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=True)
    ax.set_title("Sirbu Multi-task: C-index Across 4 Clinical Outcomes (mean ± std, 5-fold CV)",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig09_sirbu_multitask")


# ---------------------------------------------------------------------------
# Figure 10 — Feature importance (Cox + tree, average across folds)
# ---------------------------------------------------------------------------

def fig10_feature_importance(fi_records: list[dict], out_dir: Path, top_k: int = 15) -> None:
    if not fi_records:
        print("  [skip] fig10 — no feature importance data")
        return

    datasets = sorted({r["dataset"] for r in fi_records})
    for dataset in datasets:
        ds_recs = [r for r in fi_records if r["dataset"] == dataset]
        models_in_ds = sorted({r["model"] for r in ds_recs})
        if not models_in_ds:
            continue

        fig, axes = plt.subplots(1, len(models_in_ds),
                                 figsize=(len(models_in_ds) * 5.5, 7), squeeze=False)
        for ax, model in zip(axes[0], models_in_ds):
            recs = [r for r in ds_recs if r["model"] == model]
            fi_type = recs[0].get("type", "unknown")

            if fi_type == "cox_coefficients":
                # Average coef and p-value across folds
                coef_acc, p_acc = {}, {}
                for r in recs:
                    for feat, v in r.get("coef", {}).items():
                        coef_acc.setdefault(feat, []).append(v)
                    for feat, v in r.get("p", {}).items():
                        p_acc.setdefault(feat, []).append(v)
                avg_coef = {f: np.mean(vs) for f, vs in coef_acc.items()}
                avg_p    = {f: np.mean(vs) for f, vs in p_acc.items()}
                avg_hr   = {f: np.exp(v)   for f, v in avg_coef.items()}

                # Sort by |coef|, show top_k
                top_feats = sorted(avg_coef, key=lambda f: abs(avg_coef[f]), reverse=True)[:top_k]
                hrs  = [avg_hr[f] for f in top_feats]
                ps   = [avg_p.get(f, 1.0) for f in top_feats]
                colors = ["#e15759" if h > 1 else "#4e79a7" for h in hrs]
                alphas = [1.0 if p < 0.05 else 0.35 for p in ps]

                bars = ax.barh(range(len(top_feats)), [h - 1 for h in hrs], color=colors)
                for bar, a in zip(bars, alphas):
                    bar.set_alpha(a)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(range(len(top_feats)))
                ax.set_yticklabels(top_feats, fontsize=8)
                ax.set_xlabel("HR − 1  (opaque = p < 0.05)")
                ax.set_title(f"{MODEL_LABELS.get(model, model)}\nHazard Ratios", fontsize=9)

            elif fi_type == "impurity":
                imp_acc = {}
                for r in recs:
                    for feat, v in r.get("features", {}).items():
                        imp_acc.setdefault(feat, []).append(v)
                avg_imp  = {f: np.mean(vs) for f, vs in imp_acc.items()}
                top_feats = sorted(avg_imp, key=avg_imp.get, reverse=True)[:top_k]
                imps = [avg_imp[f] for f in top_feats]

                ax.barh(range(len(top_feats)), imps, color=get_model_color(model))
                ax.set_yticks(range(len(top_feats)))
                ax.set_yticklabels(top_feats, fontsize=8)
                ax.set_xlabel("Importance")
                ax.set_title(f"{MODEL_LABELS.get(model, model)}\nFeature Importance", fontsize=9)

            ax.invert_yaxis()

        fig.suptitle(f"Feature Importance — {DATASET_LABELS.get(dataset, dataset)}  "
                     f"(mean across folds)", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out_dir, f"fig10_feature_importance_{dataset.lower()}")


# ---------------------------------------------------------------------------
# Figure 11 — HPO convergence
# ---------------------------------------------------------------------------

def fig11_hpo_convergence(bp_df: pd.DataFrame, out_dir: Path) -> None:
    if bp_df.empty:
        print("  [skip] fig11 — no best_params data")
        return
    models = [m for m in MODEL_ORDER if m in bp_df["Model"].unique()]

    # 11a: best_value vs n_trials scatter
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        sub = bp_df[bp_df["Model"] == model].dropna(subset=["best_value", "n_trials"])
        if sub.empty:
            continue
        ax.scatter(sub["n_trials"], sub["best_value"],
                   color=get_model_color(model), alpha=0.65, s=65,
                   label=MODEL_LABELS.get(model, model), edgecolors="white")
    ax.set_xlabel("Number of Completed Optuna Trials")
    ax.set_ylabel("Best C-index Found by HPO")
    ax.set_title("HPO Convergence: Best C-index per Completed Trials\n"
                 "(each point = model × dataset × fold)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig11_hpo_convergence")

    # 11b: convergence speed (best_trial / n_trials)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    speed_data = []
    for model in models:
        sub = bp_df[bp_df["Model"] == model].dropna(subset=["best_trial", "n_trials"])
        if sub.empty:
            continue
        ratio = (sub["best_trial"] / sub["n_trials"]).mean()
        speed_data.append({"Model": model, "ratio": ratio,
                            "label": MODEL_LABELS.get(model, model),
                            "color": get_model_color(model)})
    if speed_data:
        sd = pd.DataFrame(speed_data).sort_values("ratio")
        ax2.barh(range(len(sd)), sd["ratio"], color=sd["color"])
        ax2.set_yticks(range(len(sd)))
        ax2.set_yticklabels(sd["label"])
        ax2.set_xlabel("Best-trial / Total-trials  (lower = faster convergence)")
        ax2.set_xlim(0, 1)
        ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.set_title("HPO Convergence Speed — Where Was the Best Trial Found?",
                      fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        for i, row in enumerate(sd.itertuples()):
            ax2.text(row.ratio + 0.01, i, f"{row.ratio:.2f}", va="center", fontsize=8)
    fig2.tight_layout()
    save_fig(fig2, out_dir, "fig11b_hpo_convergence_speed")


# ---------------------------------------------------------------------------
# Figure 12 — Model ranking
# ---------------------------------------------------------------------------

def fig12_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    models   = [m for m in MODEL_ORDER if m in df["Model"].unique()]
    datasets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]

    mean_c = (df.groupby(["Model", "Dataset"], observed=True)["C-index"]
                .mean().unstack("Dataset")
                .reindex(index=models, columns=datasets))
    ranks  = mean_c.rank(ascending=False)                   # 1 = best
    mean_ranks = ranks.mean(axis=1).dropna().sort_values()  # lower = better

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: sorted bar chart of mean rank
    ax = axes[0]
    mr_models = list(mean_ranks.index)
    colors = [get_model_color(m) for m in mr_models]
    bars = ax.barh(range(len(mr_models)), mean_ranks.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(mr_models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in mr_models])
    ax.set_xlabel("Mean Rank (↓ better)")
    ax.set_title("Average Model Rank Across All Datasets\n(ranked by C-index within each dataset)",
                 fontweight="bold")
    ax.axvline(len(models) / 2, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    for i, (m, r) in enumerate(mean_ranks.items()):
        ax.text(r + 0.05, i, f"{r:.1f}", va="center", fontsize=8.5)

    # Right: per-dataset rank heatmap
    ax = axes[1]
    rank_pivot = ranks.reindex(index=[m for m in MODEL_ORDER if m in ranks.index],
                               columns=[d for d in DATASET_ORDER if d in ranks.columns])
    ylabels = [MODEL_LABELS.get(m, m) for m in rank_pivot.index]
    xlabels = [DATASET_LABELS.get(d, d) for d in rank_pivot.columns]
    im = ax.imshow(rank_pivot.values.astype(float), cmap="RdYlGn_r",
                   vmin=1, vmax=len(models), aspect="auto")
    plt.colorbar(im, ax=ax, label="Rank (1 = best)", shrink=0.75)
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    for i in range(len(rank_pivot.index)):
        for j in range(len(rank_pivot.columns)):
            val = rank_pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("Model Rank per Dataset", fontweight="bold")

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODEL_GROUPS),
               bbox_to_anchor=(0.5, -0.04), frameon=True, title="Model Group", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, out_dir, "fig12_ranking")


# ---------------------------------------------------------------------------
# Figure 13 — D-calibration heatmap
# ---------------------------------------------------------------------------

def fig13_dcal_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    df_d = df.dropna(subset=["D-cal"])
    if df_d.empty:
        print("  [skip] fig13 — no D-cal data")
        return
    datasets = [d for d in DATASET_ORDER if d in df_d["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df_d["Model"].unique()]
    pivot = (df_d.groupby(["Model","Dataset"], observed=True)["D-cal"]
                 .mean().unstack("Dataset")
                 .reindex(index=models, columns=datasets))
    ylabels = [MODEL_LABELS.get(m, m) for m in models]
    xlabels = [DATASET_LABELS.get(d, d) for d in datasets]

    fig, ax = plt.subplots(figsize=(len(datasets) * 1.4 + 1.5, len(models) * 0.65 + 2))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="D-calibration (↑ better)", shrink=0.75)
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    for i in range(len(models)):
        for j in range(len(datasets)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                txt_col = "white" if val < 0.25 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=txt_col)
    group_separator_lines(ax, [2, 2, 4, 1, 4], "h")
    ax.set_title("D-calibration Heatmap (↑ better, mean across 5 folds)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig13_dcal_heatmap")


# ---------------------------------------------------------------------------
# Summary statistics printer (stdout for RESULTS.md)
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    metrics = ["C-index", "IBS", "AUC_mean", "D-cal"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]

    print("\n" + "=" * 90)
    print("SUMMARY: Mean ± Std across 5 folds")
    print("=" * 90)

    for dataset in [d for d in DATASET_ORDER if d in df["Dataset"].unique()]:
        print(f"\n{'─'*90}")
        print(f"  {DATASET_LABELS.get(dataset, dataset)}")
        print(f"{'─'*90}")
        sub = df[df["Dataset"] == dataset]
        agg = sub.groupby("Model", observed=True)[available].agg(["mean", "std"])
        header = f"{'Model':<22}" + "".join(f"  {m:>18}" for m in available)
        print(header)
        for model in [m for m in MODEL_ORDER if m in agg.index]:
            row = agg.loc[model]
            parts = [MODEL_LABELS.get(model, model).ljust(22)]
            for m in available:
                mu  = row[(m, "mean")]
                std = row[(m, "std")]
                if np.isnan(mu):
                    parts.append(f"  {'—':>18}")
                else:
                    parts.append(f"  {f'{mu:.3f}±{std:.3f}':>18}")
            print("".join(parts))

    # Best model per dataset
    print(f"\n{'='*90}")
    print("Best model per dataset (by C-index):")
    best = (df.groupby(["Dataset", "Model"], observed=True)["C-index"]
              .mean().groupby(level="Dataset").idxmax())
    for ds, (_, m) in best.items():
        val = df[(df["Dataset"] == ds) & (df["Model"] == m)]["C-index"].mean()
        print(f"  {DATASET_LABELS.get(ds, ds):<20} → {MODEL_LABELS.get(m, m):<20}  C-index={val:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SurvPFN comprehensive analysis — generates all figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--output-dir",  default="xai/figures", type=Path)
    parser.add_argument("--top-k",       default=15, type=int,
                        help="Max features in importance plots.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from '{results_dir}' ...")
    df = load_results(results_dir)
    print(f"  {len(df)} fold records | "
          f"{df['Model'].nunique()} models | {df['Dataset'].nunique()} datasets")

    fi_records = load_feature_importance(results_dir)
    print(f"  {len(fi_records)} feature-importance records")

    bp_df = load_best_params(results_dir)
    print(f"  {len(bp_df)} best-params records")

    print(f"\nGenerating figures → {out_dir}")
    fig01_heatmap_cindex(df, out_dir)
    fig02_cindex_comparison(df, out_dir)
    fig03_ibs_comparison(df, out_dir)
    fig04_multimetric_overview(df, out_dir)
    fig05_auc_curves(df, out_dir)
    fig06_efficiency_frontier(df, out_dir)
    fig07_model_family_boxplot(df, out_dir)
    fig08_tabpfn_ablation(df, out_dir)
    fig09_sirbu_multitask(df, out_dir)
    fig10_feature_importance(fi_records, out_dir, top_k=args.top_k)
    fig11_hpo_convergence(bp_df, out_dir)
    fig12_ranking(df, out_dir)
    fig13_dcal_heatmap(df, out_dir)

    print_summary(df)

    pdfs = list(out_dir.glob("*.pdf"))
    print(f"\n✓ Done — {len(pdfs)} figures in '{out_dir}'")


if __name__ == "__main__":
    main()
