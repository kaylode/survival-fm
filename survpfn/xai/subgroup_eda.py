"""
survpfn/xai/subgroup_eda.py — Demographic subgroup analysis for EHR datasets.

Targets MIMIC-IV (MIMIC_SURV_B) and eICU (EICU_SURV), which carry demographic
columns that allow stratification by age, sex, and race/ethnicity.

Outputs per dataset (written to <output-dir>/<dataset_name>/)
-------------------------------------------------------------
  subgroup_panel.pdf        — 3 × 3 composite: KM / event-rate bars / follow-up
                              violins for age × sex × race
  subgroup_km_age.pdf       — standalone KM, age strata
  subgroup_km_sex.pdf       — standalone KM, sex
  subgroup_km_race.pdf      — standalone KM, race/ethnicity
  subgroup_stats.csv        — per-subgroup descriptive table (machine-readable)
  subgroup_stats.tex        — LaTeX-formatted table for the appendix

Usage
-----
    uv run python -m survpfn.xai.subgroup_eda                     # eICU + MIMIC
    uv run python -m survpfn.xai.subgroup_eda --datasets EICU_SURV
    uv run python -m survpfn.xai.subgroup_eda --output-dir results/xai/subgroups
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
from scipy import stats as scipy_stats

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False
    warnings.warn(
        "lifelines not installed — KM curves will be skipped.\n"
        "Install with: uv add lifelines",
        stacklevel=1,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific demographic metadata
# ─────────────────────────────────────────────────────────────────────────────

# eICU one-hot race columns → display label
EICU_RACE_COLS = {
    "eth_caucasian":        "White",
    "eth_african_american": "Black/AA",
    "eth_hispanic":         "Hispanic",
    "eth_asian":            "Asian",
}

# MIMIC-IV one-hot race columns → display label
MIMIC_RACE_COLS = {
    "race_white":    "White",
    "race_black":    "Black",
    "race_asian":    "Asian",
    "race_hispanic": "Hispanic",
}

# MIMIC insurance columns (optional extra stratification)
MIMIC_INS_COLS = {
    "ins_medicare": "Medicare",
    "ins_medicaid": "Medicaid",
}

# Mapping dataset name → race columns
RACE_COL_MAP = {
    "EICU_SURV":   EICU_RACE_COLS,
    "MIMIC_SURV_B": MIMIC_RACE_COLS,
}

# Clinically meaningful ICU age bins
AGE_BINS   = [0, 50, 65, 75, 120]
AGE_LABELS = ["<50", "50–64", "65–74", "≥75"]

# Colour palette — consistent across datasets
_AGE_COLORS  = ["#4393c3", "#2166ac", "#92c5de", "#d1e5f0"]
_SEX_COLORS  = {"Male": "#2166ac", "Female": "#d6604d"}
_RACE_COLORS = {
    "White":      "#4dac26",
    "Black/AA":   "#d01c8b",
    "Black":      "#d01c8b",
    "Hispanic":   "#f1a340",
    "Asian":      "#998ec3",
    "Other/Unk":  "#aaaaaa",
}
_VIO_COLOR   = "#5ab4ac"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":   True,
    "grid.alpha":  0.2,
    "grid.linestyle": "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# Subgroup construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_age_groups(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a categorical age-group Series, or None if 'age' not present."""
    if "age" not in df.columns:
        return None
    return pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS,
                  right=False, include_lowest=True)


def make_sex_groups(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a 'Male'/'Female' Series, or None if 'gender_male' not present."""
    if "gender_male" not in df.columns:
        return None
    return df["gender_male"].map({1: "Male", 0: "Female", 1.0: "Male", 0.0: "Female"})


def make_race_groups(df: pd.DataFrame, dataset: str) -> Optional[pd.Series]:
    """
    Reconstruct a mutually-exclusive race/ethnicity label from one-hot columns.
    Priority: first positive column wins.  No positive → 'Other/Unk'.
    Returns None if no race columns are present.
    """
    race_cols = RACE_COL_MAP.get(dataset, {})
    present   = {col: lbl for col, lbl in race_cols.items() if col in df.columns}
    if not present:
        return None

    labels = pd.Series("Other/Unk", index=df.index, dtype="object")
    # Assign in reverse priority so first column wins (lower priority overwritten first)
    for col, lbl in reversed(list(present.items())):
        mask = df[col] == 1
        labels[mask] = lbl
    return labels


def make_insurance_groups(df: pd.DataFrame) -> Optional[pd.Series]:
    """Medicare / Medicaid / Other (MIMIC-IV only)."""
    present = {c: l for c, l in MIMIC_INS_COLS.items() if c in df.columns}
    if not present:
        return None
    labels = pd.Series("Other", index=df.index, dtype="object")
    for col, lbl in reversed(list(present.items())):
        labels[df[col] == 1] = lbl
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def logrank_p(df: pd.DataFrame, dur_col: str, ev_col: str,
              groups: pd.Series) -> float:
    """Omnibus log-rank p-value across all groups."""
    if not _HAS_LIFELINES:
        return float("nan")
    valid = groups.notna()
    try:
        res = multivariate_logrank_test(
            df.loc[valid, dur_col].astype(float),
            groups[valid].astype(str),
            df.loc[valid, ev_col].astype(int),
        )
        return float(res.p_value)
    except Exception:
        return float("nan")


def subgroup_descriptives(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    groups: pd.Series,
    label: str,
) -> pd.DataFrame:
    """
    Return a descriptive-statistics table for each subgroup level.
    Columns: Subgroup, Axis, N, Pct, EventRate, MortPct, MortCI_lo, MortCI_hi,
             MedianFU, Q25FU, Q75FU, CensoringRate
    """
    rows = []
    total = len(df)
    for grp in groups.dropna().unique():
        mask = groups == grp
        sub  = df[mask]
        n    = len(sub)
        if n < 5:
            continue
        ev   = sub[ev_col].astype(int)
        dur  = sub[dur_col].astype(float)
        k    = int(ev.sum())
        er   = k / n
        lo, hi = wilson_ci(k, n)
        rows.append({
            "Axis":          label,
            "Subgroup":      str(grp),
            "N":             n,
            "Pct_cohort":    round(100 * n / total, 1),
            "N_events":      k,
            "Event_rate":    round(er, 4),
            "Mortality_pct": round(er * 100, 1),
            "CI_lo":         round(lo * 100, 1),
            "CI_hi":         round(hi * 100, 1),
            "Median_FU":     round(float(dur.median()), 1),
            "Q25_FU":        round(float(dur.quantile(0.25)), 1),
            "Q75_FU":        round(float(dur.quantile(0.75)), 1),
            "Censoring_pct": round((1 - er) * 100, 1),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-axis figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _color_for_group(grp: str, axis: str, palette: dict | list) -> str:
    if isinstance(palette, dict):
        return palette.get(str(grp), "#888888")
    return palette[0]  # fallback


def plot_km_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    groups: pd.Series,
    axis_label: str,
    palette: dict | list,
    title: str = "",
    show_atrisk: bool = False,
) -> None:
    """
    Draw KM curves for each subgroup level on ax.
    Annotates omnibus log-rank p-value.
    """
    if not _HAS_LIFELINES:
        ax.text(0.5, 0.5, "lifelines not installed",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    p = logrank_p(df, dur_col, ev_col, groups)
    kmf = KaplanMeierFitter()
    order = sorted(groups.dropna().unique(), key=str)

    for i, grp in enumerate(order):
        mask  = (groups == grp) & groups.notna()
        color = _color_for_group(str(grp), axis_label, palette)
        n     = int(mask.sum())
        kmf.fit(df.loc[mask, dur_col].astype(float),
                event_observed=df.loc[mask, ev_col].astype(int),
                label=f"{grp} (n={n:,})")
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.12,
            color=color, linewidth=1.8,
        )

    # Log-rank p annotation
    p_str = (f"p < 0.001" if p < 0.001
             else f"p = {p:.3f}" if not np.isnan(p)
             else "p = n/a")
    ax.text(0.97, 0.97, f"log-rank {p_str}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=7.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time (hours from ICU admission)", fontsize=8)
    ax.set_ylabel("Survival probability", fontsize=8)
    ax.set_title(title or axis_label, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.8)


def plot_eventrate_axis(
    ax: plt.Axes,
    desc: pd.DataFrame,
    axis_label: str,
    palette: dict | list,
    title: str = "",
) -> None:
    """
    Horizontal bar chart of mortality % per subgroup with 95% Wilson CI.
    Annotates each bar with N.
    """
    sub = desc[desc["Axis"] == axis_label].copy()
    if sub.empty:
        ax.set_visible(False)
        return

    sub = sub.sort_values("Mortality_pct", ascending=True)
    y   = np.arange(len(sub))
    colors = [_color_for_group(g, axis_label, palette) for g in sub["Subgroup"]]

    xerr_lo = sub["Mortality_pct"].values - sub["CI_lo"].values
    xerr_hi = sub["CI_hi"].values - sub["Mortality_pct"].values

    ax.barh(y, sub["Mortality_pct"], 0.6,
            color=colors, alpha=0.82, edgecolor="white",
            xerr=[xerr_lo, xerr_hi],
            error_kw=dict(ecolor="#333333", capsize=3, elinewidth=1))

    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["CI_hi"] + 0.8,
                i, f"N={row['N']:,}", va="center", fontsize=7, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["Subgroup"], fontsize=8)
    ax.set_xlabel("Mortality rate (%)", fontsize=8)
    ax.set_title(title or axis_label, fontsize=9, fontweight="bold")
    ax.set_xlim(0, min(100, sub["CI_hi"].max() + 10))


def plot_followup_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    groups: pd.Series,
    axis_label: str,
    palette: dict | list,
    title: str = "",
) -> None:
    """
    Violin plot of follow-up hours per subgroup.
    Separately shows event (red) and censored (blue) distributions as strip points.
    """
    order  = sorted(groups.dropna().unique(), key=str)
    data   = []
    labels = []
    for grp in order:
        mask = (groups == grp) & groups.notna()
        data.append(df.loc[mask, dur_col].astype(float).values)
        labels.append(str(grp))

    if not data:
        ax.set_visible(False)
        return

    parts = ax.violinplot(data, positions=np.arange(len(data)),
                          showmedians=True, showextrema=True, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        color = _color_for_group(labels[i], axis_label, palette)
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.2)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Follow-up time (hours)", fontsize=8)
    ax.set_title(title or axis_label, fontsize=9, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(
    desc: pd.DataFrame,
    dataset_name: str,
    out_path: Path,
) -> None:
    """
    Emit a longtable with columns:
      Axis | Subgroup | N | % cohort | Mortality % (95% CI) | Median FU [IQR] hrs
    """
    latex = []
    label = dataset_name.lower().replace("_", "")
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        rf"\caption{{\textbf{{{dataset_name} — subgroup descriptive statistics.}} "
        r"Mortality\,\% with 95\,\% Wilson score CI in parentheses. "
        r"Follow-up in hours from ICU admission. "
        r"Other/Unk: patients not assigned to any specific ethnicity category.}}"
    )
    latex.append(rf"\label{{tab:subgroup_{label}}}")
    latex.append(r"\begin{tabular}{llrrrr}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Axis} & \textbf{Subgroup} & \textbf{N} "
        r"& \textbf{\% cohort} & \textbf{Mortality\,\% (95\,\%\,CI)} "
        r"& \textbf{Median FU [IQR] (h)} \\"
    )
    latex.append(r"\midrule")

    axes = desc["Axis"].unique()
    for a_idx, axis in enumerate(axes):
        sub = desc[desc["Axis"] == axis].sort_values("Subgroup")
        if a_idx > 0:
            latex.append(r"\midrule")
        latex.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{axis}}}}} \\")
        for _, row in sub.iterrows():
            fu_str = (f"{row['Median_FU']:.0f}"
                      f" [{row['Q25_FU']:.0f}--{row['Q75_FU']:.0f}]")
            ci_str = f"({row['CI_lo']:.1f}--{row['CI_hi']:.1f})"
            mort_str = f"{row['Mortality_pct']:.1f} {ci_str}"
            latex.append(
                f"  \\quad {row['Subgroup']} & & {int(row['N']):,} "
                f"& {row['Pct_cohort']:.1f} "
                f"& {mort_str} "
                f"& {fu_str} \\\\"
            )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    tex = "\n".join(latex)
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"    LaTeX table → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Composite panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_composite_panel(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    axes_cfg: list[dict],   # [{label, groups, palette, km_title, bar_title, vio_title}]
    dataset_name: str,
    out_path: Path,
) -> None:
    """
    3-row × N-col composite figure:
      Row 0: KM curves
      Row 1: Mortality-rate bar charts
      Row 2: Follow-up violins
    N = number of demographic axes (typically 3: age, sex, race).
    """
    n_axes = len(axes_cfg)
    fig, axes_grid = plt.subplots(
        3, n_axes,
        figsize=(5.5 * n_axes, 12),
        gridspec_kw={"hspace": 0.45, "wspace": 0.38},
    )
    if n_axes == 1:
        axes_grid = axes_grid.reshape(3, 1)

    # Pre-compute descriptives (all axes) for bar-chart row
    desc_all = pd.concat([
        subgroup_descriptives(df, dur_col, ev_col, cfg["groups"], cfg["label"])
        for cfg in axes_cfg
        if cfg["groups"] is not None
    ], ignore_index=True)

    for col_i, cfg in enumerate(axes_cfg):
        grp = cfg["groups"]
        pal = cfg["palette"]
        ax_km  = axes_grid[0, col_i]
        ax_bar = axes_grid[1, col_i]
        ax_vio = axes_grid[2, col_i]

        if grp is None:
            for ax in (ax_km, ax_bar, ax_vio):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_visible(False)
            continue

        plot_km_axis(ax_km, df, dur_col, ev_col, grp, cfg["label"],
                     pal, title=cfg.get("km_title", cfg["label"]))
        plot_eventrate_axis(ax_bar, desc_all, cfg["label"],
                            pal, title=cfg.get("bar_title", cfg["label"]))
        plot_followup_axis(ax_vio, df, dur_col, ev_col, grp,
                           cfg["label"], pal,
                           title=cfg.get("vio_title", cfg["label"]))

    # Add row-level annotations on the left margin
    row_labels = ["KM survival curves", "Mortality rate (95 % CI)", "Follow-up distribution"]
    for r, lbl in enumerate(row_labels):
        axes_grid[r, 0].annotate(
            lbl, xy=(0, 0.5), xytext=(-0.22, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=9, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

    fig.suptitle(
        f"{dataset_name} — Subgroup EDA  "
        f"(N={len(df):,}, mortality={df[ev_col].mean():.1%})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    composite panel → {out_path}")


def plot_standalone_km(
    df: pd.DataFrame,
    dur_col: str,
    ev_col: str,
    groups: pd.Series,
    axis_label: str,
    palette: dict | list,
    title: str,
    out_path: Path,
) -> None:
    """Standalone KM figure (larger, for individual appendix panels)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_km_axis(ax, df, dur_col, ev_col, groups, axis_label,
                 palette, title=title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    KM → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def analyse_dataset(
    name: str,
    out_dir: Path,
    include_insurance: bool = True,
) -> None:
    """Run full subgroup EDA for one EHR dataset."""
    from survpfn.dataloaders import ALL_DATASETS

    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")

    loader = ALL_DATASETS.get(name)
    if loader is None:
        print(f"  [SKIP] {name} not in registry.")
        return

    try:
        df, dur_col, ev_col = loader()
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return

    n      = len(df)
    ev_r   = df[ev_col].mean()
    print(f"  N={n:,}  mortality={ev_r:.3f}  columns={df.shape[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build subgroup series ─────────────────────────────────────────────
    age_grp  = make_age_groups(df)
    sex_grp  = make_sex_groups(df)
    race_grp = make_race_groups(df, name)
    ins_grp  = make_insurance_groups(df) if include_insurance else None

    # Summarise availability
    for label, g in [("age", age_grp), ("sex", sex_grp),
                     ("race", race_grp), ("insurance", ins_grp)]:
        if g is None:
            print(f"  [{label}] not available")
        else:
            vc = g.value_counts()
            print(f"  [{label}] groups: {dict(vc)}")

    # ── Axes configuration ────────────────────────────────────────────────
    axes_cfg: list[dict] = []

    if age_grp is not None:
        age_pal = dict(zip(AGE_LABELS, _AGE_COLORS))
        axes_cfg.append({
            "label":     "Age group",
            "groups":    age_grp,
            "palette":   age_pal,
            "km_title":  "Age group",
            "bar_title": "Age group",
            "vio_title": "Age group",
        })

    if sex_grp is not None:
        axes_cfg.append({
            "label":     "Sex",
            "groups":    sex_grp,
            "palette":   _SEX_COLORS,
            "km_title":  "Sex",
            "bar_title": "Sex",
            "vio_title": "Sex",
        })

    if race_grp is not None:
        axes_cfg.append({
            "label":     "Race/Ethnicity",
            "groups":    race_grp,
            "palette":   _RACE_COLORS,
            "km_title":  "Race/Ethnicity",
            "bar_title": "Race/Ethnicity",
            "vio_title": "Race/Ethnicity",
        })

    if ins_grp is not None and name == "MIMIC_SURV_B":
        axes_cfg.append({
            "label":     "Insurance",
            "groups":    ins_grp,
            "palette":   {"Medicare": "#1b9e77", "Medicaid": "#d95f02",
                          "Other": "#7570b3"},
            "km_title":  "Insurance type",
            "bar_title": "Insurance type",
            "vio_title": "Insurance type",
        })

    if not axes_cfg:
        print("  [WARN] no demographic axes found — skipping.")
        return

    # ── Descriptive table ─────────────────────────────────────────────────
    desc_frames = [
        subgroup_descriptives(df, dur_col, ev_col, cfg["groups"], cfg["label"])
        for cfg in axes_cfg
        if cfg["groups"] is not None
    ]
    desc = pd.concat(desc_frames, ignore_index=True)
    csv_path = out_dir / "subgroup_stats.csv"
    desc.to_csv(csv_path, index=False)
    print(f"  descriptive table → {csv_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────
    tex_path = out_dir / "subgroup_stats.tex"
    generate_latex_table(desc, name, tex_path)

    # ── Composite panel (all axes) ────────────────────────────────────────
    composite_path = out_dir / "subgroup_panel.pdf"
    plot_composite_panel(df, dur_col, ev_col, axes_cfg,
                         dataset_name=name, out_path=composite_path)

    # ── Standalone KM figures ─────────────────────────────────────────────
    for cfg in axes_cfg:
        slug = cfg["label"].lower().replace("/", "_").replace(" ", "_")
        standalone_path = out_dir / f"subgroup_km_{slug}.pdf"
        plot_standalone_km(
            df, dur_col, ev_col, cfg["groups"],
            cfg["label"], cfg["palette"],
            title=f"{name} — {cfg['km_title']}",
            out_path=standalone_path,
        )

    print(f"  Done → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATASETS = ["EICU_SURV", "MIMIC_SURV_B"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demographic subgroup EDA for EHR survival datasets (eICU, MIMIC-IV). "
            "Generates KM curves, mortality-rate bars, and follow-up violins "
            "stratified by age, sex, and race/ethnicity."
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", default=_DEFAULT_DATASETS,
        help=f"Dataset name(s) to analyse (default: {_DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/xai/subgroups"),
        help="Root output directory; one sub-folder per dataset.",
    )
    parser.add_argument(
        "--no-insurance", action="store_true",
        help="Skip insurance-type stratification for MIMIC-IV.",
    )
    args = parser.parse_args()

    print(f"\nSubgroup EDA — datasets: {args.datasets}")
    print(f"Output root:  {args.output_dir}\n")

    for name in args.datasets:
        analyse_dataset(
            name=name,
            out_dir=args.output_dir / name,
            include_insurance=not args.no_insurance,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
