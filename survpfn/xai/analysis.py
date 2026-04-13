"""
survpfn/xai/analysis.py — Comprehensive results analysis for SurvPFN.

Generates a full suite of publication-quality figures directly from the
per-fold result folders produced by scripts/benchmark.py. No aggregated
CSV is required — data is loaded on the fly.

Figures produced
----------------
fig01_heatmap_cindex         Mean C_td heatmap (models × datasets)
fig02_cindex_comparison      C_td bar charts grouped by dataset
fig03_ibs_comparison         IBS bar charts (lower is better)
fig04_multimetric_overview   Box plots: C_td / IBS / AUC / D-cal
fig05_auc_curves             Time-dependent AUROC per dataset
fig06_efficiency_frontier    C_td vs training time (log-scale scatter)
fig07_model_family_boxplot   C_td box plots per model group
fig08_tabpfn_ablation        TabPFN variants vs classical counterparts
fig09_ormoni_tirodei_multitask Performance across the 4 OrmoniTirodei outcomes
fig10_feature_importance_*   Cox hazard ratios + tree importances per dataset
fig11_hpo_convergence        Optuna best-value vs n_trials
fig11b_hpo_convergence_speed Optuna convergence speed (best trial / n_trials)
fig12a_mean_rank             Average model rank across all datasets (bar chart)
fig12b_rank_heatmap          Model rank per dataset (heatmap)
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
    "deepsurv", "mtlr", "pchazard", "deephit_single", "survtrace", "soden",

    # ── Strategy 2 — Jointly-trained / Specialized adapters ────────────────
    "tabpfn_joint_cox",
    "tabdpt_joint_cox",
    "tabicl_joint_cox",

    "tabicl_joint_cox",

    # ── Strategy 3 — Zero-shot / In-context ────────────────────────────────
    # "tabpfn_zeroshot",
    # "tabdpt_zeroshot",
    # "tabicl_zeroshot",

    # "tabpfn_zeroshot_perbin",
    # "tabdpt_zeroshot_perbin",
    # "tabicl_zeroshot_perbin",

    # "tabpfn_zeroshot_perbin_time",
    # "tabdpt_zeroshot_perbin_time",
    # "tabicl_zeroshot_perbin_time",

    "tabpfn_zeroshot_perbin_time_ens",
    "tabdpt_zeroshot_perbin_time_ens",
    "tabicl_zeroshot_perbin_time_ens",

    # ── Strategy 4 & 5 — Finetuning & TabTune ──────────────────────────────
    "tabpfn_finetune",
    "tabdpt_finetune",
    "tabicl_finetune",

    # ── Strategy 1 — Embedding-based ───────────────────────────────────────
    "tabpfn_embedding_pchazard",
    "tabdpt_embedding_pchazard",
    "tabicl_embedding_pchazard",
    "tabpfn_embedding_mtlr",
    "tabdpt_embedding_mtlr",
    "tabicl_embedding_mtlr",
    "tabpfn_embedding_deephit",
    "tabdpt_embedding_deephit",
    "tabicl_embedding_deephit",
    "tabpfn_embedding_cox",
    "tabdpt_embedding_cox",
    "tabicl_embedding_cox",

    # ── Competing Risks (CR) — Grouped by Strategy ─────────────────────────
    "cox_cr", "deephit_cr", "aj_cr", "fine_gray_cr",

    # CR — Embedding-based
    "tabpfn_embedding_deephit_cr", "tabdpt_embedding_deephit_cr", "tabicl_embedding_deephit_cr",
    "tabpfn_embedding_deephit_v2_cr", "tabdpt_embedding_deephit_v2_cr", "tabicl_embedding_deephit_v2_cr",
    "tabpfn_embedding_deephit_v2_cr_adapter", "tabdpt_embedding_deephit_v2_cr_adapter", "tabicl_embedding_deephit_v2_cr_adapter",
    "tabpfn_embedding_cox_cr", "tabdpt_embedding_cox_cr", "tabicl_embedding_cox_cr",
    "tabpfn_embedding_cox_cr_adapter", "tabdpt_embedding_cox_cr_adapter", "tabicl_embedding_cox_cr_adapter",

    # CR — Jointly-trained
    "tabpfn_joint_deephit_cr", "tabdpt_joint_deephit_cr", "tabicl_joint_deephit_cr",
    "tabpfn_joint_deephit_v2_cr", "tabdpt_joint_deephit_v2_cr", "tabicl_joint_deephit_v2_cr",
    "tabpfn_joint_deephit_v2_cr_adapter", "tabdpt_joint_deephit_v2_cr_adapter", "tabicl_joint_deephit_v2_cr_adapter",
    "tabpfn_joint_cox_cr", "tabdpt_joint_cox_cr", "tabicl_joint_cox_cr",
    "tabpfn_joint_cox_cr_adapter", "tabdpt_joint_cox_cr_adapter", "tabicl_joint_cox_cr_adapter",

    # CR — Zero-shot
    "tabpfn_zeroshot_cr_multinomial", "tabpfn_zeroshot_cr_perevent",
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
    "survtrace":      "SurvTrace",
    "soden":          "SODEN",
    "beta_surv":      "Beta-Surv",

    "tabpfn_zeroshot": "TabPFN-ZS",
    "tabdpt_zeroshot": "TabDPT-ZS",
    "tabicl_zeroshot": "TabICL-ZS",

    "tabpfn_zeroshot_perbin": "TabPFN-ZS-PB",
    "tabdpt_zeroshot_perbin": "TabDPT-ZS-PB",
    "tabicl_zeroshot_perbin": "TabICL-ZS-PB",

    "tabpfn_zeroshot_perbin_time_ens": "TabPFN-ZS", #-PB-Time-Ens",
    "tabdpt_zeroshot_perbin_time_ens": "TabDPT-ZS",#-PB-Time-Ens",
    "tabicl_zeroshot_perbin_time_ens": "TabICL-ZS",#-PB-Time-Ens",

    "tabpfn_zeroshot_perbin_time": "TabPFN-ZS-PB-Time",
    "tabdpt_zeroshot_perbin_time": "TabDPT-ZS-PB-Time",
    "tabicl_zeroshot_perbin_time": "TabICL-ZS-PB-Time",

    "tabpfn_surv_adapter": "TabPFN-Surv-Adapter",
    "tabdpt_surv_adapter": "TabDPT-Surv-Adapter",
    "tabicl_surv_adapter": "TabICL-Surv-Adapter",

    "tabpfn_tabtune": "TabPFN-TabTune",
    "tabdpt_tabtune": "TabDPT-TabTune",
    "tabicl_tabtune": "TabICL-TabTune",

    "tabpfn_new": "TabPFN-New",
    "tabpfn_finetune": "TabPFN-FT-CE",
    "tabdpt_finetune": "TabDPT-FT-CE",
    "tabicl_finetune": "TabICL-FT-CE",

    "cox_cr":          "Cox-CR",
    "deephit_cr":      "DeepHit-CR",
    "tabpfn_zeroshot_cr_multinomial": "TabPFN-ZS-CR-Mul",
    "tabpfn_zeroshot_cr_perevent":    "TabPFN-ZS-CR-PE",
}

# Compatibility mapping for older joint naming
for b in ["tabpfn", "tabdpt", "tabicl"]:
    for h in ["cox", "deephit", "pchazard", "mtlr"]:
        MODEL_LABELS[f"{b}_{h}"] = MODEL_LABELS.get(f"{b}_joint_{h}", b.upper() + "-Joint-" + h.capitalize())

# Dynamic FM labels
for fm in ["tabpfn", "tabdpt", "tabicl"]:
    fm_name = fm.replace("tab", "Tab").replace("pfn", "PFN").replace("dpt", "DPT").replace("icl", "ICL")
    for task in ["embedding", "joint"]:
        task_name = "FT" if task == "embedding" else "Joint"
        for head in ["cox", "deephit", "pchazard", "mtlr"]:
            h_name = {"cox": "Cox", "deephit": "DeepHit", "pchazard": "PCHaz", "mtlr": "MTLR"}[head]
            mid = f"{fm}_{task}_{head}"
            MODEL_LABELS[mid] = f"{fm_name}-{task_name}-{h_name}"
            MODEL_LABELS[f"{mid}_adapter"] = f"{fm_name}-{task_name}-{h_name}+Adp"

    # CR variants
    for task in ["embedding", "joint"]:
        task_name = "FT" if task == "embedding" else "Joint"
        for head in ["deephit_cr", "deephit_v2_cr", "cox_cr"]:
            h_name = {"deephit_cr": "DH-CR", "deephit_v2_cr": "DH-v2-CR", "cox_cr": "Cox-CR"}[head]
            mid = f"{fm}_{task}_{head}"
            MODEL_LABELS[mid] = f"{fm_name}-{task_name}-{h_name}"
            MODEL_LABELS[f"{mid}_adapter"] = f"{fm_name}-{task_name}-{h_name}+Adp"

MODEL_GROUPS = {
    "Baseline": ["km", "cox"],
    "Tree":     ["rsf", "gbsa"],
    "Deep":     ["deepsurv", "mtlr", "pchazard", "deephit_single", "survtrace", "soden"],

    "Finetune-Cox":      [m for m in MODEL_ORDER if "_embedding" in m and "_cox" in m and "_cr" not in m],
    "Finetune-DeepHit":   [m for m in MODEL_ORDER if "_embedding" in m and "deephit" in m and "_cr" not in m],
    "Finetune-MTLR":      [m for m in MODEL_ORDER if "_embedding" in m and "mtlr" in m and "_cr" not in m],
    "Finetune-PCHazard":  [m for m in MODEL_ORDER if "_embedding" in m and "pchazard" in m and "_cr" not in m],
    
    "Joint-Cox":      [m for m in MODEL_ORDER if ("_joint" in m or m == "tabpfn_new" or m == "tabpfn_cox") and ("_cox" in m or m == "tabpfn_new" or m == "tabpfn_cox") and "_cr" not in m],
    "Joint-DeepHit":  [m for m in MODEL_ORDER if "_joint" in m and "deephit" in m and "_cr" not in m],
    "Joint-MTLR":     [m for m in MODEL_ORDER if "_joint" in m and "mtlr" in m and "_cr" not in m],
    "Joint-PCHazard": [m for m in MODEL_ORDER if "_joint" in m and "pchazard" in m and "_cr" not in m],
    
    # "Zero-shot":            ["tabpfn_zeroshot", "tabdpt_zeroshot", "tabicl_zeroshot", "tabpfn_zeroshot_perbin", "tabdpt_zeroshot_perbin", "tabicl_zeroshot_perbin"],
    # "Zero-shot (Temporal)": ["tabpfn_zeroshot_perbin_time", "tabdpt_zeroshot_perbin_time", "tabicl_zeroshot_perbin_time", "tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens"],
    "Zero-shot": ["tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens"],
    "Finetune-CE": ["tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune"],

    "Deep-CR":       ["cox_cr", "deephit_cr", "aj_cr", "fine_gray_cr"],
    "FM-CR-Cox":     [m for m in MODEL_ORDER if "_cr" in m and "cox" in m and m not in {"cox_cr", "deephit_cr", "aj_cr", "fine_gray_cr"}],
    "FM-CR-DeepHit": [m for m in MODEL_ORDER if "_cr" in m and "deephit" in m and m not in {"cox_cr", "deephit_cr", "aj_cr", "fine_gray_cr"}],
}
MODEL_TO_GROUP = {m: g for g, ms in MODEL_GROUPS.items() for m in ms}

GROUP_COLORS = {
    "Baseline": "#4e79a7",
    "Tree":     "#f28e2b",
    "Deep":     "#59a14f",

    "Finetune-Cox":      "#e15759",
    "Finetune-DeepHit":  "#76b6b2",
    "Finetune-MTLR":     "#17becf",
    "Finetune-PCHazard": "#f1ce63",

    # "Joint-Cox":      "#9467bd",
    # "Joint-DeepHit":  "#c5b0d5",
    # "Joint-MTLR":     "#17becf",
    # "Joint-PCHazard": "#dbdb8d",
    # "Surv-Adapter":   "#8c564b",

    "Zero-shot":            "#ff9da7",
    # "Zero-shot (Temporal)": "#ff7f0e",

    "Finetune-CE": "#e377c2",
    # "TabTune":  "#e377c2",

    # "Deep-CR":       "#8c564b",
    # "FM-CR-Cox":     "#76b6b2",
    # "FM-CR-DeepHit": "#6fa8dc",
}


# --- SurvSet Benchmark Subset ---
SURVSET_BENCHMARK = [
    "cancer", "breast", "GBSG2", "rott2", "colon", "prostate", "ovarian", "Melanoma",
    "e1684", "pbc", "hepatoCellular", "nwtco", "retinopathy", "heart", "veteran",
    "whas500", "mgus", "cgd", "cost", "LeukSurv", "Dialysis", "actg", "rhc", "vlbw",
    "grace", "TRACE", "support2", "DLBCL", "diabetes", "flchain", "Framingham",
]

DATASET_ORDER = [
    # Core Public (SR)
    "SUPPORT2", "METABRIC", "GBSG", "FLCHAIN", "VETERANS", "WHAS500", "SEER",
    # ORMONI_TIRODEI (SR)
    "ORMONI_TIRODEI_MORTALITY", "ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI", "ORMONI_TIRODEI_STROKE", "ORMONI_TIRODEI",
    "EICU_SURV", "MIMIC_SURV_B",
    # Competing Risks
    "FRAMINGHAM", "PBC2", "SUPPORT_CR", "SYNTHETIC_CR",
] + ["SS_" + k.upper() for k in SURVSET_BENCHMARK]

DATASET_LABELS = {
    "SUPPORT2":       "SUPPORT2",
    "METABRIC":       "METABRIC",
    "GBSG":           "GBSG",
    "FLCHAIN":        "FL-Chain",
    "VETERANS":       "Veterans",
    "WHAS500":        "WHAS500",
    "SEER":           "SEER",
    "ORMONI_TIRODEI":          "OrmoniTirodei",
    "ORMONI_TIRODEI_MORTALITY":"Orm.Tir.-Mort.",
    "ORMONI_TIRODEI_CV":       "Orm.Tir.-CV",
    "ORMONI_TIRODEI_MI":       "Orm.Tir.-MI",
    "ORMONI_TIRODEI_STROKE":   "Orm.Tir.-Stroke",
    "FRAMINGHAM":     "Fram-CR",
    "PBC2":           "PBC2-CR",
    "SUPPORT_CR":     "Supp-CR",
    "SYNTHETIC_CR":   "Syn-CR",
    "EICU_SURV" : "eICU", 
    "MIMIC_SURV_B": "MIMIC-IV",

}
for k in SURVSET_BENCHMARK:
    DATASET_LABELS["SS_" + k.upper()] = f"SS-{k.capitalize()}"

PUBLIC_DATASETS = ["EICU_SURV", "MIMIC_SURV_B", "SUPPORT2", "METABRIC", "GBSG", "FLCHAIN", "VETERANS", "WHAS500", "SEER"] + ["SS_" + k.upper() for k in SURVSET_BENCHMARK]
ORMONI_TIRODEI_DATASETS  = ["ORMONI_TIRODEI_MORTALITY", "ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI", "ORMONI_TIRODEI_STROKE", "ORMONI_TIRODEI"]
CR_DATASETS     = ["FRAMINGHAM", "PBC2", "SUPPORT_CR", "SYNTHETIC_CR"]
SR_DATASETS     = [d for d in DATASET_ORDER if d not in CR_DATASETS]


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
        
        try:
            with metrics_path.open() as f:
                metrics = json.load(f)
        except:
            print(f"Error loading metrics from {metrics_path}")
            exit(0)

        meta = {}
        meta_path = metrics_path.parent / "metadata.json"
        if meta_path.exists():
            try:
                with meta_path.open() as f:
                    meta = json.load(f)
            except:
                print(f"Error loading metadata from {meta_path}")
                exit(0)

        record = {"Dataset": dataset, "Model": model, "Fold": fold}
        record.update(metrics)
        for key in ("fit_time_s", "eval_time_s", "n_params",
                    "n_train", "n_test", "n_features",
                    "n_events_train", "n_events_test",
                    "event_rate_train", "event_rate_test"):
            record[key] = meta.get(key)
        records.append(record)
    if len(records) == 0: return None
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
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
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
# Figure 01 — C_td heatmap
# ---------------------------------------------------------------------------

def fig01_heatmap_cindex(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in df["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df["Model"].unique()]
    pivot = (df.groupby(["Model", "Dataset"], observed=True)["C_td"]
               .mean()
               .unstack("Dataset")
               .reindex(index=models, columns=datasets))

    fig, ax = plt.subplots(figsize=(len(datasets) * 1.4 + 1.5, len(models) * 0.65 + 2))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", vmin=0.45, vmax=0.85, aspect="auto")
    plt.colorbar(im, ax=ax, label="C_td (↑ better)", shrink=0.75)

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

    # Group dividers
    groups_present = []
    current_group = None
    group_counts = []
    for m in models:
        g = MODEL_TO_GROUP.get(m, "Unknown")
        if g != current_group:
            groups_present.append(g)
            group_counts.append(1)
            current_group = g
        else:
            group_counts[-1] += 1
    
    group_separator_lines(ax, group_counts, "h")

    # Add group labels on the left
    y = -0.5
    for g, count in zip(groups_present, group_counts):
        label_y = y + count / 2
        ax.text(-0.8, label_y, g, ha="right", va="center", fontsize=7.5,
                color=GROUP_COLORS.get(g, "black"), fontweight="bold")
        y += count

    ax.set_title("Mean C_td Across 5-fold CV", fontweight="bold", pad=12)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig01_heatmap_cindex")


# ---------------------------------------------------------------------------
# Figure 02 — C_td bar charts per dataset
# ---------------------------------------------------------------------------

def fig02_cindex_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in SR_DATASETS if d in df["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df["Model"].unique()]

    n_cols = min(3, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols

    #. Drop nan
    df = df[df['C_td'].notna()]

    summary = (df.groupby(["Dataset", "Model"], observed=True)["C_td"]
                 .agg(["mean", "std"]).reset_index())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(models).dropna(subset=["mean"]))
        if sub.empty:
            ax.set_visible(False)
            continue

        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        if "cox" in sub.index:
            ax.axhline(sub.loc["cox", "mean"], color="#4e79a7",
                       linestyle="--", linewidth=1, alpha=0.6, label="Cox baseline")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=90, ha="right", fontsize=8)
        ax.set_ylabel("C_td" if idx % n_cols == 0 else "")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        ymax = min(1.0, sub["mean"].max() + 0.08)
        ax.set_ylim(0.3, ymax)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODEL_GROUPS),
               bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("C_td Comparison (mean ± std, 5-fold CV)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig02_cindex_comparison")


def fig02_auc_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in SR_DATASETS if d in df["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df["Model"].unique()]

    n_cols = min(3, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols

    summary = (df.groupby(["Dataset", "Model"], observed=True)["AUC_mean"]
                 .agg(["mean", "std"]).reset_index())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(models).dropna(subset=["mean"]))
        if sub.empty:
            ax.set_visible(False)
            continue

        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=90, ha="right", fontsize=8)
        ax.set_ylabel("AUC (↑ better)" if idx % n_cols == 0 else "")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(MODEL_GROUPS),
               bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("Mean AUC Comparison (mean ± std, 5-fold CV)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig02_auc_comparison")


# ---------------------------------------------------------------------------
# Figure 03 — IBS bar charts
# ---------------------------------------------------------------------------

def fig03_ibs_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    df_ibs = df.dropna(subset=["IBS"])
    if df_ibs.empty:
        print("  [skip] fig03 — no IBS data")
        return
    datasets = [d for d in SR_DATASETS if d in df_ibs["Dataset"].unique()]
    models   = [m for m in MODEL_ORDER   if m in df_ibs["Model"].unique()]

    n_cols = min(3, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    summary = (df_ibs.groupby(["Dataset", "Model"], observed=True)["IBS"]
                     .agg(["mean", "std"]).reset_index())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(models).dropna(subset=["mean"]))
        if sub.empty:
            ax.set_visible(False)
            continue

        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        if "cox" in sub.index:
            ax.axhline(sub.loc["cox", "mean"], color="#4e79a7",
                       linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=90, ha="right", fontsize=8)
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


def fig03_auc_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Exactly like fig02_auc but using fig03 naming convention as requested."""
    fig02_auc_comparison(df, out_dir) # Reuse logic


# ---------------------------------------------------------------------------
# Figure 04 — Multi-metric box plots (all datasets pooled)
# ---------------------------------------------------------------------------

def fig04_multimetric_overview(df: pd.DataFrame, out_dir: Path) -> None:
    metrics_cfg = [
        ("C_td",  "C_td (↑)",  (0.3, 1.0)),
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
                           rotation=90, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(label, fontweight="bold")
        # Group separators
        current_g = None
        pos = 0.5
        for m in models:
            g = MODEL_TO_GROUP.get(m, "Unknown")
            if current_g is not None and g != current_g:
                ax.axvline(pos, color="gray", linestyle=":", alpha=0.4)
            current_g = g
            pos += 1

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
    sub = df.dropna(subset=["C_td", "fit_time_s"])
    if sub.empty:
        print("  [skip] fig06 — no timing data")
        return
    agg = (sub.groupby(["Model", "Dataset", "Group"], observed=True)
              .agg(cindex=("C_td", "mean"), fit_time=("fit_time_s", "mean"))
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
    ax.set_ylabel("Mean C_td")
    ax.set_title("Performance–Efficiency Frontier\n(each point = model × dataset mean)",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig06_efficiency_frontier")


# ---------------------------------------------------------------------------
# Figure 07 — Model family box plots (public vs OrmoniTirodei)
# ---------------------------------------------------------------------------

def fig07_model_family_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    groups = list(MODEL_GROUPS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (split_datasets, title) in zip(axes, [
        ([d for d in PUBLIC_DATASETS if d in df["Dataset"].unique()], "Public Datasets"),
        ([d for d in ORMONI_TIRODEI_DATASETS  if d in df["Dataset"].unique()], "OrmoniTirodei (Private)"),
    ]):
        if not split_datasets:
            ax.set_visible(False)
            continue
        sub = df[df["Dataset"].isin(split_datasets)].copy()
        sub["Group"] = sub["Model"].map(MODEL_TO_GROUP)
        data   = [sub[sub["Group"] == g]["C_td"].dropna().values for g in groups]
        colors = [GROUP_COLORS[g] for g in groups]

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=4, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=15)
        ax.set_ylabel("C_td")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        for i, (g, d) in enumerate(zip(groups, data), 1):
            ax.text(i, 0.31, f"n={len(d)}", ha="center", fontsize=7, color="gray")

    fig.suptitle("C_td Distribution by Model Family (Public vs Private)",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig07_model_family_boxplot")


# ---------------------------------------------------------------------------
# Figure 08 — TabPFN ablation
# ---------------------------------------------------------------------------

def fig08_tabpfn_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("cox",           "tabpfn_embedding_cox", "Cox PH",   "TabPFN-Emb-Cox"),
        ("cox",           "tabpfn_cox",           "Cox PH",   "TabPFN-Joint-Cox"),
        ("mtlr",          "tabpfn_mtlr",          "MTLR",     "TabPFN-Joint-MTLR"),
        ("deephit_single","tabpfn_deephit",       "DeepHit",  "TabPFN-Joint-DeepHit"),
        ("pchazard",      "tabpfn_pchazard",      "PC-Haz",   "TabPFN-Joint-PCHaz"),
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
                      .groupby("Dataset", observed=True)["C_td"].agg(["mean","std"]))
        tabpfn_agg = (df[df["Model"] == tabpfn_m]
                      .groupby("Dataset", observed=True)["C_td"].agg(["mean","std"]))
        ds = [d for d in datasets if d in base_agg.index or d in tabpfn_agg.index]
        x = np.arange(len(ds)); w = 0.35

        b_m  = [base_agg.loc[d,"mean"]   if d in base_agg.index   else np.nan for d in ds]
        b_s  = [base_agg.loc[d,"std"]    if d in base_agg.index   else 0       for d in ds]
        t_m  = [tabpfn_agg.loc[d,"mean"] if d in tabpfn_agg.index else np.nan for d in ds]
        t_s  = [tabpfn_agg.loc[d,"std"]  if d in tabpfn_agg.index else 0       for d in ds]

        ax.bar(x - w/2, b_m, w, yerr=b_s, label=base_l,
               color=get_model_color(base_m), capsize=3, edgecolor="white")
        tabpfn_col = (GROUP_COLORS["Joint-Cox"] if tabpfn_m.startswith("tabpfn_")
                      else GROUP_COLORS["Embedding-Cox"])
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
        ax.set_ylabel("C_td")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"{base_l}  →  {tabpfn_l}", fontweight="bold")

    fig.suptitle("TabPFN Ablation: Classical vs Foundation Model Counterparts\n"
                 "(Δ C_td shown above bars; green = improvement, red = degradation)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig08_tabpfn_ablation")


# ---------------------------------------------------------------------------
# Figure 09 — OrmoniTirodei multi-task
# ---------------------------------------------------------------------------

def fig09_ormoni_tirodei_multitask(df: pd.DataFrame, out_dir: Path) -> None:
    """Radar plot or faceted bar chart comparing models across all 4 OrmoniTirodei outcomes."""
    datasets = ["ORMONI_TIRODEI_MORTALITY", "ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI", "ORMONI_TIRODEI_STROKE"]
    ormoni_tirodei_ds = [d for d in datasets if d in df["Dataset"].unique()]
    if not ormoni_tirodei_ds:
        print("  [skip] fig09 — no OrmoniTirodei multi-task data")
        return
    models = [m for m in MODEL_ORDER
              if m in df[df["Dataset"].isin(ormoni_tirodei_ds)]["Model"].unique()
              and m != "km"]

    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(ormoni_tirodei_ds)
    x = np.arange(n)
    w = 0.7 / len(models)
    offsets = np.linspace(-0.35 + w/2, 0.35 - w/2, len(models))

    for model, offset in zip(models, offsets):
        sub = df[df["Dataset"].isin(ormoni_tirodei_ds) & (df["Model"] == model)]
        means = sub.groupby("Dataset", observed=True)["C_td"].mean().reindex(ormoni_tirodei_ds)
        stds  = sub.groupby("Dataset", observed=True)["C_td"].std().reindex(ormoni_tirodei_ds)
        ax.bar(x + offset, means, w * 0.92, yerr=stds,
               label=MODEL_LABELS.get(model, model),
               color=get_model_color(model), capsize=2,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in ormoni_tirodei_ds])
    ax.set_ylabel("C_td")
    ax.set_ylim(0.3, 0.92)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=True)
    fig.suptitle("Performance Consistency across OrmoniTirodei Outcomes", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig09_ormoni_tirodei_multitask")


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
    ax.set_ylabel("Best C_td Found by HPO")
    ax.set_title("HPO Convergence: Best C_td per Completed Trials\n"
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

    configs = [
        ("C_td", False, "fig12a_mean_rank", "fig12b_rank_heatmap", 
         "Average Model Rank Across All Datasets\n(ranked by C_td within each dataset)",
         "Model Rank per Dataset"),
        ("IBS", True, "fig12c_mean_rank_ibs", "fig12d_rank_heatmap_ibs",
         "Average Model Rank Across All Datasets\n(ranked by IBS within each dataset, lowest=1)",
         "Model Rank per Dataset (IBS)"),
        ("AUC_mean", False, "fig12i_mean_rank_auc", "fig12j_rank_heatmap_auc",
         "Average Model Rank Across All Datasets\n(ranked by Mean AUC)",
         "Model Rank per Dataset (Mean AUC)")
    ]

    for metric, asc, fname_mean, fname_heat, title_mean, title_heat in configs:
        if metric not in df.columns or df[metric].isna().all():
            continue

        mean_val = (df.groupby(["Model", "Dataset"], observed=True)[metric]
                    .mean().unstack("Dataset")
                    .reindex(index=models, columns=datasets))
        ranks  = mean_val.rank(ascending=asc)                   # 1 = best
        mean_ranks = ranks.mean(axis=1) .dropna().sort_values() # lower = better

        # --- Mean Rank Bar Chart ---
        fig_a, ax_a = plt.subplots(figsize=(10, 7))
        mr_models = list(mean_ranks.index)
        colors = [get_model_color(m) for m in mr_models]
        ax_a.barh(range(len(mr_models)), mean_ranks.values, color=colors, edgecolor="white")
        ax_a.set_yticks(range(len(mr_models)))
        ax_a.set_yticklabels([MODEL_LABELS.get(m, m) for m in mr_models])
        ax_a.set_xlabel("Mean Rank (↓ better)")
        ax_a.set_title(title_mean, fontweight="bold")
        ax_a.axvline(len(models) / 2, color="gray", linestyle="--", alpha=0.5)
        ax_a.grid(axis="x", alpha=0.3)
        for i, (m, r) in enumerate(mean_ranks.items()):
            ax_a.text(r + 0.05, i, f"{r:.1f}", va="center", fontsize=8.5)

        legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
        fig_a.legend(handles=legend_handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
        fig_a.tight_layout(rect=[0, 0.05, 1, 1])
        save_fig(fig_a, out_dir, fname_mean)

        # --- Per-dataset Rank Heatmap ---
        fig_b, ax_b = plt.subplots(figsize=(11, 8))
        rank_pivot = ranks.reindex(index=[m for m in MODEL_ORDER if m in ranks.index],
                                   columns=[d for d in DATASET_ORDER if d in ranks.columns])
        ylabels = [MODEL_LABELS.get(m, m) for m in rank_pivot.index]
        xlabels = [DATASET_LABELS.get(d, d) for d in rank_pivot.columns]
        im = ax_b.imshow(rank_pivot.values.astype(float), cmap="RdYlGn_r",
                        vmin=1, vmax=len(models), aspect="auto")
        plt.colorbar(im, ax=ax_b, label="Rank (1 = best)", shrink=0.75)
        ax_b.set_xticks(range(len(xlabels))); ax_b.set_xticklabels(xlabels, rotation=35, ha="right")
        ax_b.set_yticks(range(len(ylabels))); ax_b.set_yticklabels(ylabels)
        for i in range(len(rank_pivot.index)):
            for j in range(len(rank_pivot.columns)):
                val = rank_pivot.iloc[i, j]
                if not np.isnan(val):
                    ax_b.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
        ax_b.set_title(title_heat, fontweight="bold")

        fig_b.legend(handles=legend_handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
        fig_b.tight_layout(rect=[0, 0.05, 1, 1])
        save_fig(fig_b, out_dir, fname_heat)


def fig12_ranking_cr(df: pd.DataFrame, out_dir: Path) -> None:
    """Version of Fig 12 specifically for Competing Risks results."""
    # Filter for CR datasets
    datasets = [d for d in CR_DATASETS if d in df["Dataset"].unique()]
    if not datasets:
        print("  [skip] fig12_cr — no CR datasets in provided dataframe")
        return

    # Filter for models that have results in these datasets
    present_models = df[df["Dataset"].isin(datasets)]["Model"].unique()
    models = [m for m in MODEL_ORDER if m in present_models]

    configs = [
        ("C_td", False, "fig12e_mean_rank_cr", "fig12f_rank_heatmap_cr", 
         "Average Model Rank (Competing Risks)\n(ranked by Macro C_td)",
         "Model Rank per Dataset (CR)"),
        ("IBS", True, "fig12g_mean_rank_ibs_cr", "fig12h_rank_heatmap_ibs_cr",
         "Average Model Rank (Competing Risks)\n(ranked by Macro IBS, lowest=1)",
         "Model Rank per Dataset (CR IBS)"),
        ("AUC_mean", False, "fig12k_mean_rank_auc_cr", "fig12l_rank_heatmap_auc_cr",
         "Average Model Rank (Competing Risks)\n(ranked by Macro Mean AUC)",
         "Model Rank per Dataset (CR Mean AUC)")
    ]

    for metric, asc, fname_mean, fname_heat, title_mean, title_heat in configs:
        if metric not in df.columns or df[metric].isna().all():
            continue

        mean_val = (df.groupby(["Model", "Dataset"], observed=True)[metric]
                    .mean().unstack("Dataset")
                    .reindex(index=models, columns=datasets))
        
        # Drop models that have NO data in any of the CR datasets
        mean_val = mean_val.dropna(how="all", axis=0)
        
        ranks  = mean_val.rank(ascending=asc)                   # 1 = best
        mean_ranks = ranks.mean(axis=1).dropna().sort_values() # lower = better

        # --- Mean Rank Bar Chart ---
        fig_a, ax_a = plt.subplots(figsize=(10, 7))
        mr_models = list(mean_ranks.index)
        colors = [get_model_color(m) for m in mr_models]
        ax_a.barh(range(len(mr_models)), mean_ranks.values, color=colors, edgecolor="white")
        ax_a.set_yticks(range(len(mr_models)))
        ax_a.set_yticklabels([MODEL_LABELS.get(m, m) for m in mr_models])
        ax_a.set_xlabel("Mean Rank (↓ better)")
        ax_a.set_title(title_mean, fontweight="bold")
        ax_a.axvline(len(mr_models) / 2, color="gray", linestyle="--", alpha=0.5)
        ax_a.grid(axis="x", alpha=0.3)
        for i, (m, r) in enumerate(mean_ranks.items()):
            ax_a.text(r + 0.05, i, f"{r:.1f}", va="center", fontsize=8.5)

        legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                          if any(MODEL_TO_GROUP.get(m) == g for m in mr_models)]
        fig_a.legend(handles=legend_handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
        fig_a.tight_layout(rect=[0, 0.05, 1, 1])
        save_fig(fig_a, out_dir, fname_mean)

        # --- Per-dataset Rank Heatmap ---
        fig_b, ax_b = plt.subplots(figsize=(11, 8))
        rank_pivot = ranks.reindex(index=mr_models[::-1], columns=datasets)
        ylabels = [MODEL_LABELS.get(m, m) for m in rank_pivot.index]
        xlabels = [DATASET_LABELS.get(d, d) for d in rank_pivot.columns]
        im = ax_b.imshow(rank_pivot.values.astype(float), cmap="RdYlGn_r",
                        vmin=1, vmax=len(mr_models), aspect="auto")
        plt.colorbar(im, ax=ax_b, label="Rank (1 = best)", shrink=0.75)
        ax_b.set_xticks(range(len(xlabels))); ax_b.set_xticklabels(xlabels, rotation=35, ha="right")
        ax_b.set_yticks(range(len(ylabels))); ax_b.set_yticklabels(ylabels)
        for i in range(len(rank_pivot.index)):
            for j in range(len(rank_pivot.columns)):
                val = rank_pivot.iloc[i, j]
                if not np.isnan(val):
                    ax_b.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
        ax_b.set_title(title_heat, fontweight="bold")

        fig_b.legend(handles=legend_handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
        fig_b.tight_layout(rect=[0, 0.05, 1, 1])
        save_fig(fig_b, out_dir, fname_heat)


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
    # Group dividers
    groups_present = []
    current_group = None
    group_counts = []
    for m in models:
        g = MODEL_TO_GROUP.get(m, "Unknown")
        if g != current_group:
            groups_present.append(g)
            group_counts.append(1)
            current_group = g
        else:
            group_counts[-1] += 1
    
    group_separator_lines(ax, group_counts, "h")

    ax.set_title("D-calibration Heatmap (↑ better, mean across 5 folds)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig13_dcal_heatmap")


# ---------------------------------------------------------------------------
# Figure 14 & 15 — Competing Risk Comparisons
# ---------------------------------------------------------------------------

def fig14_cindex_comparison_cr(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in CR_DATASETS if d in df["Dataset"].unique()]
    if not datasets:
        print("  [skip] fig14 — no CR dataset results")
        return
    # Focus only on models that support CR (+ some generic baselines)
    cr_models = [m for m in MODEL_ORDER if m in df["Model"].unique() 
                 and ("_cr" in m or m == "deephit_cr")]
    
    n_cols = min(2, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    summary = (df.groupby(["Dataset", "Model"], observed=True)["C_td"]
                 .agg(["mean", "std"]).reset_index())
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4.5), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(cr_models).dropna(subset=["mean"]))
        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        if "cox_cr" in sub.index:
            ax.axhline(sub.loc["cox_cr", "mean"], color="#8c564b",
                       linestyle="--", linewidth=1, alpha=0.6, label="Cox-CR baseline")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("C_td")
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(6, len(MODEL_GROUPS)),
               bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("Competing Risks: C_td Comparison (Macro-average)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig14_cindex_comparison_cr")


def fig15_ibs_comparison_cr(df: pd.DataFrame, out_dir: Path) -> None:
    df_ibs = df.dropna(subset=["IBS"])
    datasets = [d for d in CR_DATASETS if d in df_ibs["Dataset"].unique()]
    if not datasets:
        print("  [skip] fig15 — no CR IBS results")
        return
    cr_models = [m for m in MODEL_ORDER if m in df_ibs["Model"].unique()
                 and ("_cr" in m or m == "deephit_cr")]
    
    n_cols = min(2, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    summary = (df_ibs.groupby(["Dataset", "Model"], observed=True)["IBS"]
                     .agg(["mean", "std"]).reset_index())
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4.5), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = (summary[summary["Dataset"] == dataset]
               .set_index("Model").reindex(cr_models).dropna(subset=["mean"]))
        colors = [get_model_color(m) for m in sub.index]
        ax.bar(range(len(sub)), sub["mean"], yerr=sub["std"],
               color=colors, capsize=3, width=0.7, edgecolor="white", linewidth=0.4)
        if "cox_cr" in sub.index:
            ax.axhline(sub.loc["cox_cr", "mean"], color="#8c564b",
                       linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sub.index],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("IBS (↓ better)")
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(6, len(MODEL_GROUPS)),
               bbox_to_anchor=(0.5, -0.05), frameon=True, title="Model Group", fontsize=9)
    fig.suptitle("Competing Risks: IBS Comparison (Macro-average, ↓ is Better)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig15_ibs_comparison_cr")


def fig16_pareto_ci_ibs(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter plot of Mean C_td vs Mean Integrated Brier Score (IBS)
    with a Pareto frontier (maximizing CI, minimizing IBS).
    """
    sub = df.dropna(subset=["C_td", "IBS"])
    if sub.empty:
        print("  [skip] fig16 — no CI or IBS data")
        return

    agg = (sub.groupby(["Model", "Group"], observed=True)
              .agg(cindex=("C_td", "mean"), ibs=("IBS", "mean"))
              .reset_index()
              .dropna())

    if agg.empty:
        print("  [skip] fig16 — aggregated data is empty")
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for group, color in GROUP_COLORS.items():
        g = agg[agg["Group"] == group]
        if g.empty: continue
        ax.scatter(g["cindex"], g["ibs"], color=color, s=120, 
                   alpha=0.8, label=group, edgecolors="white", linewidth=0.5, zorder=3)

    for _, row in agg.iterrows():
        ax.annotate(MODEL_LABELS.get(row["Model"], row["Model"]),
                    (row["cindex"], row["ibs"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points", alpha=0.9)

    # Pareto calculation
    srt_desc = agg.sort_values("cindex", ascending=False)
    pareto_points = []
    curr_min_ibs = np.inf
    for _, row in srt_desc.iterrows():
        if row["ibs"] < curr_min_ibs:
            curr_min_ibs = row["ibs"]
            pareto_points.append((row["cindex"], row["ibs"]))
    
    if pareto_points:
        pareto_points = sorted(pareto_points, key=lambda x: x[0])
        px = [p[0] for p in pareto_points]
        py = [p[1] for p in pareto_points]
        ax.plot(px, py, color="crimson", linewidth=2.5, linestyle="--", 
                alpha=0.8, label="Pareto Frontier", zorder=5)
        ax.scatter(px, py, color="crimson", s=150, marker="*", edgecolors="white", zorder=6)

    ax.set_xlabel("Mean C_td (↑ better)", fontsize=11)
    ax.set_ylabel("Mean Integrated Brier Score (↓ better)", fontsize=11)
    ax.set_title("Pareto Frontier: Performance (C_td) vs Calibration (IBS)\n"
                 "(Average across all datasets)", fontweight="bold", fontsize=13)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig16_pareto_ci_ibs")



# ---------------------------------------------------------------------------
# Summary statistics printer (stdout for RESULTS.md)
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    metrics = ["C_td", "IBS", "AUC_mean", "D-cal"]
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
    print("Best model per dataset (by C_td):")
    best_means = df.groupby(["Dataset", "Model"], observed=True)["C_td"].mean().dropna()
    if not best_means.empty:
        best = best_means.groupby(level="Dataset", observed=True).idxmax()
        for ds, (_, m) in best.items():
            val = df[(df["Dataset"] == ds) & (df["Model"] == m)]["C_td"].mean()
            print(f"  {DATASET_LABELS.get(ds, ds):<20} → {MODEL_LABELS.get(m, m):<20}  C_td={val:.4f}")
    else:
        print("  No valid C_td data found.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SurvPFN comprehensive analysis — generates all figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results/benchmark", type=Path)
    parser.add_argument("--results-dir-cr", default="results/benchmark_cr", type=Path)
    parser.add_argument("--output-dir",  default="results/xai/figures", type=Path)
    parser.add_argument("--top-k",       default=15, type=int,
                        help="Max features in importance plots.")
    parser.add_argument("--figures",     nargs="+", default=["all"],
                        help="Specific figures to run (e.g. 01 02 10). Use 'all' for everything.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir_cr = Path(args.results_dir_cr)
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
    print_summary(df)
    
    
    print(f"Loading CR results from '{results_dir_cr}' ...")
    df_cr = load_results(results_dir_cr)
    if df_cr is not None:
        print(f"  {len(df_cr)} fold records | "
            f"{df_cr['Model'].nunique()} models | {df_cr['Dataset'].nunique()} datasets")
        bp_df_cr = load_best_params(results_dir_cr)
        print(f"  {len(bp_df_cr)} best-params records (CR)")
        fi_records_cr = load_feature_importance(results_dir_cr)
        print(f"  {len(fi_records_cr)} feature-importance records (CR)")

        print_summary(df_cr)

    # Map figure IDs to functions
    fig_map = {
        "01": lambda: fig01_heatmap_cindex(df, out_dir),
        "02": lambda: fig02_cindex_comparison(df, out_dir),
        "02_auc": lambda: fig02_auc_comparison(df, out_dir),
        "03": lambda: fig03_ibs_comparison(df, out_dir),
        "03_auc": lambda: fig03_auc_comparison(df, out_dir),
        "04": lambda: fig04_multimetric_overview(df, out_dir),
        "05": lambda: fig05_auc_curves(df, out_dir),
        "06": lambda: fig06_efficiency_frontier(df, out_dir),
        "07": lambda: fig07_model_family_boxplot(df, out_dir),
        "08": lambda: fig08_tabpfn_ablation(df, out_dir),
        "09": lambda: fig09_ormoni_tirodei_multitask(df, out_dir),
        "10": lambda: fig10_feature_importance(fi_records, out_dir, top_k=args.top_k),
        "11": lambda: fig11_hpo_convergence(bp_df, out_dir),
        "12": lambda: fig12_ranking(df, out_dir),
        "12_cr": lambda: fig12_ranking_cr(df_cr, out_dir),
        "13": lambda: fig13_dcal_heatmap(df, out_dir),
        "14": lambda: fig14_cindex_comparison_cr(df_cr, out_dir),
        "15": lambda: fig15_ibs_comparison_cr(df_cr, out_dir),
        "16": lambda: fig16_pareto_ci_ibs(df, out_dir),
    }

    if "all" in args.figures:
        to_run = sorted(fig_map.keys())
    else:
        to_run = []
        for f in args.figures:
            fid = f.lower().replace("fig", "").zfill(2)
            if fid in fig_map:
                to_run.append(fid)
            else:
                print(f"  [skip] Figure '{f}' (ID: {fid}) not recognized.")
   
    if not to_run:
        print("No figures selected. Exiting.")
        return

    print(f"\nGenerating {len(to_run)} figures → {out_dir}")
    for fid in sorted(to_run):
        fig_map[fid]()


    pdfs = list(out_dir.glob("*.pdf"))
    print(f"\n✓ Done — {len(pdfs)} figures in '{out_dir}'")


if __name__ == "__main__":
    main()
