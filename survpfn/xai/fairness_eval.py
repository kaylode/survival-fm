"""
survpfn/xai/fairness_eval.py — Fairness evaluation across demographic subgroups.

Loads per-fold risk-score predictions saved by ``benchmark.py --save-predictions``,
joins them with raw demographic columns from the original dataset, computes
C-index per (model × fold × subgroup), aggregates across folds, and generates
four figure types designed for a paper appendix.

Demographic axes (auto-detected from column presence)
------------------------------------------------------
  Age group  : <50 / 50–64 / 65–74 / ≥75    (column: ``age``)
  Sex        : Male / Female                  (column: ``gender_male``)
  Race/Ethn. : White / Black/AA / Hispanic / Asian / Other  (one-hot columns)
  Insurance  : Medicare / Medicaid / Other   (MIMIC-IV only)

Figures (saved to <output-dir>/<dataset>/)
------------------------------------------
  fairness_heatmap_age.pdf    — model × age-group C-index heatmap (mean ± std)
  fairness_heatmap_sex.pdf    — model × sex C-index heatmap
  fairness_heatmap_race.pdf   — model × race C-index heatmap
  fairness_heatmap_ins.pdf    — model × insurance C-index heatmap (MIMIC only)
  fairness_gap.pdf            — C-index gap (max−min group) per model, per axis
  fairness_grouped_bars.pdf   — absolute C-index per group for key model families
  fairness_delta.pdf          — Δ(subgroup C − overall C) per model per group
  fairness_results.csv        — machine-readable full results
  fairness_results.tex        — LaTeX table for appendix

Usage
-----
    # Run after: benchmark.py --datasets EICU_SURV MIMIC_SURV_B --save-predictions
    uv run python -m survpfn.xai.fairness_eval
    uv run python -m survpfn.xai.fairness_eval --datasets EICU_SURV MIMIC_SURV_B
    uv run python -m survpfn.xai.fairness_eval \\
        --pred-dir   results/predictions \\
        --output-dir results/xai/fairness \\
        --min-n      30
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from sksurv.metrics import concordance_index_censored
    _HAS_SKSURV = True
except ImportError:
    _HAS_SKSURV = False
    try:
        from lifelines.utils import concordance_index as _li_cindex
        _HAS_LIFELINES = True
    except ImportError:
        _HAS_LIFELINES = False
        warnings.warn(
            "Neither sksurv nor lifelines found — C-index computation unavailable.\n"
            "Install with: uv add scikit-survival  OR  uv add lifelines",
            stacklevel=1,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Subgroup construction — reuse logic from subgroup_eda.py
# ─────────────────────────────────────────────────────────────────────────────

from survpfn.xai.subgroup_eda import (
    make_age_groups,
    make_sex_groups,
    make_race_groups,
    make_insurance_groups,
    AGE_BINS,
    AGE_LABELS,
    _AGE_COLORS,
    _SEX_COLORS,
    _RACE_COLORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Axis metadata  (label → display name, colour palette)
# ─────────────────────────────────────────────────────────────────────────────

_INS_COLORS = {"Medicare": "#1b9e77", "Medicaid": "#d95f02", "Other": "#7570b3"}

AXES_META = {
    "Age group":     {"palette": dict(zip(AGE_LABELS, _AGE_COLORS))},
    "Sex":           {"palette": _SEX_COLORS},
    "Race/Ethnicity":{"palette": _RACE_COLORS},
    "Insurance":     {"palette": _INS_COLORS},
}

# Model-family colour for gap / delta plots
MODEL_FAMILY_COLORS = {
    "classical": "#2166ac",
    "tree":      "#4dac26",
    "deep":      "#d6604d",
    "fm":        "#8856a7",
}

_MODEL_FAMILY = {
    "cox":            "classical", "km":            "classical",
    "rsf":            "tree",      "gbsa":          "tree",
    "deepsurv":       "deep",      "mtlr":          "deep",
    "pchazard":       "deep",      "deephit_single":"deep",
    "survtrace":      "deep",      "soden":         "deep",
    "beta_surv":      "fm",
}

def _family(model: str) -> str:
    for prefix, fam in _MODEL_FAMILY.items():
        if model == prefix or model.startswith(prefix + "_"):
            return fam
    if any(k in model for k in ("tabpfn", "tabdpt", "tabicl")):
        return "fm"
    return "deep"


# ─────────────────────────────────────────────────────────────────────────────
# C-index computation
# ─────────────────────────────────────────────────────────────────────────────

def _cindex(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index. risk is higher-is-worse (event happens sooner)."""
    mask = np.isfinite(risk) & np.isfinite(duration)
    if mask.sum() < 10:
        return float("nan")
    d, e, r = duration[mask], event[mask].astype(bool), risk[mask]
    try:
        if _HAS_SKSURV:
            c, _, _, _, _ = concordance_index_censored(e, d, r)
        else:
            # lifelines: predicted_score should be HIGHER for longer survivors
            c = _li_cindex(d, -r, e)
        return float(c)
    except Exception:
        return float("nan")


def compute_subgroup_cindex(
    risk: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    groups: pd.Series,
    min_n: int = 20,
) -> dict[str, float]:
    """Return C-index for each subgroup level. Groups with < min_n are NaN."""
    out: dict[str, float] = {}
    for grp in groups.dropna().unique():
        mask = (groups == grp).values
        if mask.sum() < min_n:
            out[str(grp)] = float("nan")
            continue
        out[str(grp)] = _cindex(duration[mask], event[mask], risk[mask])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load & aggregate predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_fold_predictions(pred_dir: Path, dataset: str, model: str, fold: int
                          ) -> Optional[pd.DataFrame]:
    path = pred_dir / dataset / model / f"fold_{fold}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def discover_models(pred_dir: Path, dataset: str) -> list[str]:
    """Return sorted list of model names found under pred_dir/dataset/."""
    base = pred_dir / dataset
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def compute_dataset_results(
    pred_dir: Path,
    dataset: str,
    raw_df: pd.DataFrame,
    axes: list[dict],        # [{label, groups_series}]
    min_n: int,
    models: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For every (model, fold), compute per-subgroup C-index.
    Returns a long-form DataFrame with columns:
        model, fold, axis, subgroup, c_index, n_test, n_subgroup
    """
    avail_models = models or discover_models(pred_dir, dataset)
    records = []

    for model in avail_models:
        model_dir = pred_dir / dataset / model
        fold_files = sorted(model_dir.glob("fold_*.parquet")) if model_dir.exists() else []
        if not fold_files:
            continue

        for fold_path in fold_files:
            fold = int(fold_path.stem.split("_")[1])
            preds = pd.read_parquet(fold_path)

            # Join with raw demographics via original_idx
            preds = preds.merge(
                raw_df.reset_index(drop=False).rename(columns={"index": "original_idx"}),
                on="original_idx", how="left",
            )

            risk = preds["risk_score"].values
            dur  = preds["duration"].values
            ev   = preds["event"].values

            # Overall C-index (no subgroup filter)
            c_overall = _cindex(dur, ev, risk)
            records.append({
                "model": model, "fold": fold,
                "axis": "Overall", "subgroup": "Overall",
                "c_index": c_overall, "n_test": len(preds), "n_subgroup": len(preds),
            })

            # Per-axis subgroup C-index
            for ax in axes:
                label  = ax["label"]
                grp_fn = ax["group_fn"]
                groups = grp_fn(preds)
                if groups is None:
                    continue
                sub_results = compute_subgroup_cindex(risk, dur, ev, groups, min_n=min_n)
                vc = groups.value_counts()
                for grp_label, c in sub_results.items():
                    records.append({
                        "model": model, "fold": fold,
                        "axis": label, "subgroup": grp_label,
                        "c_index": c,
                        "n_test": len(preds),
                        "n_subgroup": int(vc.get(grp_label, 0)),
                    })

    return pd.DataFrame(records)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold C-index → mean ± std across folds.
    Returns: model, axis, subgroup, mean_c, std_c, n_folds, mean_n_subgroup
    """
    out = (
        df.groupby(["model", "axis", "subgroup"])
        .agg(
            mean_c=("c_index", "mean"),
            std_c=("c_index", "std"),
            n_folds=("fold", "nunique"),
            mean_n=("n_subgroup", "mean"),
        )
        .reset_index()
    )
    out["std_c"] = out["std_c"].fillna(0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_CMAP_FAIRNESS = "RdYlGn"   # red = low C, green = high C


def _model_display(m: str) -> str:
    """Shorten model names for axis labels."""
    return (m.replace("embedding_", "emb-")
             .replace("_embedding", "-emb")
             .replace("tabpfn", "TabPFN")
             .replace("tabdpt", "TabDPT")
             .replace("tabicl", "TabICL")
             .replace("deephit_single", "DeepHit")
             .replace("deepsurv", "DeepSurv")
             .replace("pchazard", "PCH")
             .replace("beta_surv", "BetaSurv")
             .replace("zeroshot", "ZS"))


# ── Figure 1: C-index heatmap ─────────────────────────────────────────────

def plot_heatmap(
    agg: pd.DataFrame,
    axis_label: str,
    dataset: str,
    out_path: Path,
    vmin: float = 0.45,
    vmax: float = 0.85,
) -> None:
    """
    Heatmap: rows = models (sorted by overall C-index),
             cols = subgroup levels,
             colour = mean C-index, annotated with "mean\n±std".
    """
    overall = (
        agg[agg["axis"] == "Overall"][["model", "mean_c"]]
        .rename(columns={"mean_c": "overall_c"})
    )
    sub = agg[agg["axis"] == axis_label].copy()
    if sub.empty:
        return

    sub = sub.merge(overall, on="model", how="left")
    sub = sub.sort_values("overall_c", ascending=False)

    models   = sub["model"].unique().tolist()  # preserve sort order
    models   = list(dict.fromkeys(sub["model"].tolist()))  # stable deduplicated order
    subgroups = sorted(sub["subgroup"].unique().tolist())

    # Build matrix
    mat_mean = np.full((len(models), len(subgroups)), np.nan)
    mat_std  = np.full((len(models), len(subgroups)), np.nan)
    for i, m in enumerate(models):
        for j, sg in enumerate(subgroups):
            row = sub[(sub["model"] == m) & (sub["subgroup"] == sg)]
            if not row.empty:
                mat_mean[i, j] = row["mean_c"].values[0]
                mat_std[i, j]  = row["std_c"].values[0]

    n_m, n_sg = len(models), len(subgroups)
    fig, ax = plt.subplots(figsize=(max(4, n_sg * 1.3), max(3, n_m * 0.45)))

    im = ax.imshow(mat_mean, cmap=_CMAP_FAIRNESS, vmin=vmin, vmax=vmax,
                   aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="C-index")

    # Annotate cells
    for i in range(n_m):
        for j in range(n_sg):
            val = mat_mean[i, j]
            std = mat_std[i, j]
            if np.isnan(val):
                txt = "—"
                color = "gray"
            else:
                txt = f"{val:.3f}\n±{std:.3f}"
                color = "white" if (val < vmin + 0.2 * (vmax - vmin)
                                    or val > vmin + 0.8 * (vmax - vmin)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    ax.set_xticks(range(n_sg))
    ax.set_xticklabels(subgroups, fontsize=8)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([_model_display(m) for m in models], fontsize=7)
    ax.set_title(f"{dataset} — C-index by {axis_label}\n"
                 f"(sorted by overall C-index, mean ± std across folds)",
                 fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    heatmap → {out_path}")


# ── Figure 2: Performance gap chart ─────────────────────────────────────────

def plot_gap_chart(
    agg: pd.DataFrame,
    axes_labels: list[str],
    dataset: str,
    out_path: Path,
) -> None:
    """
    For each demographic axis: horizontal bar showing max−min C-index across
    subgroups per model.  Sorted by gap descending. Colour = model family.
    Models with all-NaN subgroups are excluded.
    """
    n_axes = len(axes_labels)
    fig, axs = plt.subplots(1, n_axes, figsize=(5 * n_axes, max(4, len(agg["model"].unique()) * 0.35)),
                             sharey=True)
    if n_axes == 1:
        axs = [axs]

    # Sort models by mean overall C-index
    overall_order = (
        agg[agg["axis"] == "Overall"]
        .groupby("model")["mean_c"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for ax, axis_label in zip(axs, axes_labels):
        sub = agg[agg["axis"] == axis_label].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        gaps = (
            sub.groupby("model")["mean_c"]
            .agg(lambda x: np.nanmax(x) - np.nanmin(x))
            .reindex(overall_order)
            .dropna()
        )
        gaps = gaps.sort_values(ascending=True)  # ascending so worst gap is at bottom

        colors = [MODEL_FAMILY_COLORS.get(_family(m), "#888888") for m in gaps.index]
        y = np.arange(len(gaps))

        ax.barh(y, gaps.values, 0.65, color=colors, alpha=0.82, edgecolor="white")
        ax.axvline(gaps.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean = {gaps.mean():.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels([_model_display(m) for m in gaps.index], fontsize=7)
        ax.set_xlabel("C-index gap (max − min group)", fontsize=8)
        ax.set_title(f"{axis_label}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Family legend (on first axis)
    handles = [mpatches.Patch(color=c, label=f.capitalize())
               for f, c in MODEL_FAMILY_COLORS.items()]
    axs[0].legend(handles=handles, title="Model family", fontsize=7,
                  title_fontsize=8, loc="lower right")

    fig.suptitle(f"{dataset} — Fairness gap: C-index (max−min across groups)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    gap chart → {out_path}")


# ── Figure 3: Grouped bar chart (key models × subgroups) ────────────────────

def plot_grouped_bars(
    agg: pd.DataFrame,
    axis_label: str,
    dataset: str,
    out_path: Path,
    key_models: Optional[list[str]] = None,
) -> None:
    """
    X = subgroup level, grouped bars = key model representatives (one per family).
    Shows absolute C-index per group, making cross-model comparisons within groups clear.
    """
    sub = agg[agg["axis"] == axis_label].copy()
    if sub.empty:
        return

    # Auto-select one representative per family if not specified
    if key_models is None:
        overall_c = (
            agg[agg["axis"] == "Overall"]
            .groupby("model")["mean_c"].mean()
        )
        key_models = []
        for fam in ("classical", "tree", "deep", "fm"):
            fam_models = [m for m in overall_c.index if _family(m) == fam]
            if fam_models:
                best = max(fam_models, key=lambda m: overall_c.get(m, 0))
                key_models.append(best)

    key_models = [m for m in key_models if m in sub["model"].unique()]
    if not key_models:
        return

    subgroups = sorted(sub["subgroup"].unique())
    n_sg = len(subgroups)
    n_m  = len(key_models)
    width = 0.8 / n_m

    family_colors = [MODEL_FAMILY_COLORS.get(_family(m), "#888888") for m in key_models]

    fig, ax = plt.subplots(figsize=(max(5, n_sg * 1.1), 4.5))
    x = np.arange(n_sg)

    for i, (model, color) in enumerate(zip(key_models, family_colors)):
        means = []
        errs  = []
        for sg in subgroups:
            row = sub[(sub["model"] == model) & (sub["subgroup"] == sg)]
            if row.empty or np.isnan(row["mean_c"].values[0]):
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(row["mean_c"].values[0])
                errs.append(row["std_c"].values[0])

        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9, color=color, alpha=0.82,
               label=_model_display(model), edgecolor="white",
               yerr=errs, error_kw=dict(ecolor="#333333", capsize=2, elinewidth=0.8))

    # Overall C-index reference lines per model
    for model, color in zip(key_models, family_colors):
        overall_row = agg[(agg["model"] == model) & (agg["axis"] == "Overall")]
        if not overall_row.empty:
            ax.axhline(overall_row["mean_c"].values[0], color=color,
                       linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, fontsize=8)
    ax.set_ylabel("C-index (mean ± std across folds)", fontsize=9)
    ax.set_title(f"{dataset} — C-index by {axis_label}\n"
                 f"(dashed = overall C-index per model)", fontsize=9)
    ax.legend(title="Model", fontsize=7, title_fontsize=8, ncol=2)
    ax.set_ylim(max(0, ax.get_ylim()[0] - 0.02), min(1, ax.get_ylim()[1] + 0.02))
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    grouped bars → {out_path}")


# ── Figure 4: Delta from overall ─────────────────────────────────────────────

def plot_delta(
    agg: pd.DataFrame,
    axis_label: str,
    dataset: str,
    out_path: Path,
) -> None:
    """
    Per-model: Δ = (subgroup C-index − overall C-index).
    X = models sorted by overall C-index.
    One coloured line per subgroup group.
    Positive Δ = model performs *better* than average for that group.
    Negative Δ = model is *worse* for that group.
    """
    sub = agg[agg["axis"] == axis_label].copy()
    if sub.empty:
        return

    overall = (
        agg[agg["axis"] == "Overall"][["model", "mean_c"]]
        .rename(columns={"mean_c": "overall_c"})
    )
    sub = sub.merge(overall, on="model", how="left")
    sub["delta"] = sub["mean_c"] - sub["overall_c"]

    model_order = (
        agg[agg["axis"] == "Overall"]
        .groupby("model")["mean_c"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    model_order = [m for m in model_order if m in sub["model"].unique()]
    if not model_order:
        return

    subgroups = sorted(sub["subgroup"].unique())

    # Get colour palette for this axis
    pal = AXES_META.get(axis_label, {}).get("palette", {})

    fig, ax = plt.subplots(figsize=(max(6, len(model_order) * 0.55), 4.5))
    x = np.arange(len(model_order))

    for sg in subgroups:
        sg_data = sub[sub["subgroup"] == sg].set_index("model")
        y = [sg_data.loc[m, "delta"] if m in sg_data.index else np.nan
             for m in model_order]
        color = pal.get(sg, "#888888")
        ax.plot(x, y, marker="o", markersize=5, linewidth=1.5,
                color=color, label=sg, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.fill_between([-0.5, len(model_order) - 0.5], -0.02, 0,
                    color="#d6604d", alpha=0.06, zorder=0)
    ax.fill_between([-0.5, len(model_order) - 0.5], 0, 0.02,
                    color="#4dac26", alpha=0.06, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_display(m) for m in model_order],
                       rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Δ C-index (subgroup − overall)", fontsize=9)
    ax.set_title(f"{dataset} — Bias pattern: Δ C-index by {axis_label}\n"
                 f"(models sorted by overall C-index, left = best)",
                 fontsize=9)
    ax.legend(title=axis_label, fontsize=7, title_fontsize=8, loc="best",
              framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_xlim(-0.5, len(model_order) - 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    delta plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(agg: pd.DataFrame, dataset: str, out_path: Path) -> None:
    """
    Compact LaTeX table: model × subgroup C-index (mean ± std) for each axis.
    One block per demographic axis, separated by \\midrule.
    """
    label = dataset.lower().replace("_", "")
    overall = (
        agg[agg["axis"] == "Overall"][["model", "mean_c"]]
        .rename(columns={"mean_c": "overall_c"})
        .set_index("model")
    )
    model_order = overall["overall_c"].sort_values(ascending=False).index.tolist()

    axes_labels = [a for a in agg["axis"].unique() if a != "Overall"]

    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"\centering\small")
    latex.append(
        rf"\caption{{\textbf{{{dataset} — fairness evaluation: C-index per subgroup.}} "
        r"Mean $\pm$ std across folds. "
        r"Models sorted by overall C-index (descending). "
        r"Grey cells indicate fewer than the minimum sample threshold. "
        r"Cells with bold values indicate the highest C-index within each column.}}"
    )
    latex.append(rf"\label{{tab:fairness_{label}}}")

    # Determine max subgroup count per axis for column sizing
    all_subgroups: dict[str, list[str]] = {}
    for ax in axes_labels:
        sgs = sorted(agg[agg["axis"] == ax]["subgroup"].unique())
        all_subgroups[ax] = sgs

    # Build a flat column list: Overall | [Age subgroups] | [Sex subgroups] | [Race subgroups]
    col_defs = "l" + "r" * (1 + sum(len(v) for v in all_subgroups.values()))
    latex.append(rf"\begin{{tabular}}{{{col_defs}}}")
    latex.append(r"\toprule")

    # Header row 1: axis spans
    header1 = r"\textbf{Model} & \textbf{Overall}"
    for ax, sgs in all_subgroups.items():
        header1 += rf" & \multicolumn{{{len(sgs)}}}{{c}}{{\textbf{{{ax}}}}}"
    latex.append(header1 + r" \\")

    # Header row 2: subgroup labels
    header2 = " & "
    subgroup_cols = []
    for ax, sgs in all_subgroups.items():
        for sg in sgs:
            subgroup_cols.append((ax, sg))
            header2 += rf" & \textit{{{sg}}}"
    latex.append(header2 + r" \\")
    latex.append(r"\midrule")

    for model in model_order:
        disp = _model_display(model)
        ov = overall.loc[model, "overall_c"] if model in overall.index else float("nan")
        row = rf"\texttt{{{disp}}} & {ov:.3f}"
        for ax, sg in subgroup_cols:
            cell = agg[(agg["model"] == model) & (agg["axis"] == ax) & (agg["subgroup"] == sg)]
            if cell.empty or np.isnan(cell["mean_c"].values[0]):
                row += r" & \textcolor{gray}{--}"
            else:
                m, s = cell["mean_c"].values[0], cell["std_c"].values[0]
                row += f" & {m:.3f}$\\pm${s:.3f}"
        latex.append(row + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"    LaTeX → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def analyse_dataset(
    dataset: str,
    pred_dir: Path,
    out_dir: Path,
    min_n: int = 30,
    models: Optional[list[str]] = None,
    key_models: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """Run full fairness evaluation for one dataset. Returns aggregated results."""
    from survpfn.dataloaders import ALL_DATASETS

    print(f"\n{'═'*60}")
    print(f"  {dataset}")
    print(f"{'═'*60}")

    # Check predictions exist
    if not (pred_dir / dataset).exists():
        print(f"  [SKIP] No predictions found at {pred_dir / dataset}")
        print(f"         Run benchmark.py --save-predictions --datasets {dataset} first.")
        return None

    # Load raw dataset (for demographics — unscaled)
    loader = ALL_DATASETS.get(dataset)
    if loader is None:
        print(f"  [SKIP] {dataset} not in registry.")
        return None
    try:
        raw_df, dur_col, ev_col = loader()
        # Keep only demographic columns to avoid join conflicts with prediction cols
        demo_cols = ["age", "gender_male",
                     "eth_caucasian", "eth_african_american", "eth_hispanic", "eth_asian",
                     "race_white", "race_black", "race_asian", "race_hispanic",
                     "ins_medicare", "ins_medicaid"]
        keep = [c for c in demo_cols if c in raw_df.columns]
        raw_df = raw_df[keep].reset_index()
        print(f"  Raw dataset: {len(raw_df):,} rows | demo columns: {keep}")
    except Exception as exc:
        print(f"  [FAIL] Could not load raw dataset: {exc}")
        return None

    # Build demographic axes
    dummy = raw_df  # raw_df has the demographic cols
    axes = []
    ag = make_age_groups(dummy)
    if ag is not None:
        axes.append({"label": "Age group",
                     "group_fn": make_age_groups})
    sg = make_sex_groups(dummy)
    if sg is not None:
        axes.append({"label": "Sex",
                     "group_fn": make_sex_groups})
    rg = make_race_groups(dummy, dataset)
    if rg is not None:
        axes.append({"label": "Race/Ethnicity",
                     "group_fn": lambda df: make_race_groups(df, dataset)})
    ig = make_insurance_groups(dummy)
    if ig is not None:
        axes.append({"label": "Insurance",
                     "group_fn": make_insurance_groups})

    if not axes:
        print("  [WARN] No demographic axes found.")
        return None

    print(f"  Axes: {[ax['label'] for ax in axes]}")

    # Compute per-fold subgroup C-indices
    print("  Computing subgroup C-indices per fold ...")
    fold_df = compute_dataset_results(
        pred_dir=pred_dir, dataset=dataset, raw_df=raw_df,
        axes=axes, min_n=min_n, models=models,
    )

    if fold_df.empty:
        print("  [SKIP] No prediction data found.")
        return None

    print(f"  {len(fold_df['model'].unique())} models × {len(fold_df['fold'].unique())} folds")

    # Aggregate
    agg = aggregate_results(fold_df)

    # Save CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fairness_results_folds.csv", index=False)
    agg.to_csv(out_dir / "fairness_results.csv", index=False)
    print(f"  Results → {out_dir / 'fairness_results.csv'}")

    axes_labels = [ax["label"] for ax in axes]

    # ── Generate figures ──────────────────────────────────────────────────

    # 1. Heatmaps
    for ax_label in axes_labels:
        slug = ax_label.lower().replace("/", "_").replace(" ", "_")
        plot_heatmap(agg, ax_label, dataset, out_dir / f"fairness_heatmap_{slug}.pdf")

    # 2. Gap chart (all axes in one figure)
    plot_gap_chart(agg, axes_labels, dataset, out_dir / "fairness_gap.pdf")

    # 3. Grouped bars (one per axis)
    for ax_label in axes_labels:
        slug = ax_label.lower().replace("/", "_").replace(" ", "_")
        plot_grouped_bars(agg, ax_label, dataset, out_dir / f"fairness_bars_{slug}.pdf",
                          key_models=key_models)

    # 4. Delta from overall (one per axis)
    for ax_label in axes_labels:
        slug = ax_label.lower().replace("/", "_").replace(" ", "_")
        plot_delta(agg, ax_label, dataset, out_dir / f"fairness_delta_{slug}.pdf")

    # 5. LaTeX table
    generate_latex_table(agg, dataset, out_dir / "fairness_results.tex")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATASETS = ["EICU_SURV", "MIMIC_SURV_B"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fairness evaluation: per-subgroup C-index across demographic groups "
            "for EHR survival models.  Requires benchmark.py to have been run "
            "with --save-predictions."
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", default=_DEFAULT_DATASETS,
        help=f"Datasets to evaluate (default: {_DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--pred-dir", type=Path, default=Path("results/predictions"),
        help="Root directory of saved predictions (default: results/predictions).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/xai/fairness"),
        help="Root output directory; one sub-folder per dataset.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict to these model names (default: all found in pred-dir).",
    )
    parser.add_argument(
        "--key-models", nargs="+", default=None,
        help="Models to highlight in grouped-bar figures (default: auto, best per family).",
    )
    parser.add_argument(
        "--min-n", type=int, default=30,
        help="Minimum subgroup size to compute C-index (default: 30).",
    )
    args = parser.parse_args()

    print(f"\nFairness evaluation")
    print(f"  Datasets:  {args.datasets}")
    print(f"  Pred dir:  {args.pred_dir}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Min N:     {args.min_n}")

    for dataset in args.datasets:
        analyse_dataset(
            dataset=dataset,
            pred_dir=args.pred_dir,
            out_dir=args.output_dir / dataset,
            min_n=args.min_n,
            models=args.models,
            key_models=args.key_models,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
