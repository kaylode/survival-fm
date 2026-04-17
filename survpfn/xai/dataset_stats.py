"""
survpfn/xai/dataset_stats.py — Dataset statistics, diversity plots, and LaTeX table.

Statistics computed per dataset
--------------------------------
  N, p (features), event_rate, censoring_rate,
  median_followup, iqr_followup, max_followup,
  n_numeric, n_categorical, n_binary, missing_pct, mean_abs_corr,
  n_bins (adaptive), n_causes (CR only),
  domain, regime, type (SR | CR), cite, authors

Diversity figures (saved to <output-dir>/figures/)
---------------------------------------------------
  diversity_scatter.pdf  — log N vs event rate, bubble size = features, colour = domain
  feature_composition.pdf — stacked bar of continuous / binary / categorical features
  censoring_vs_followup.pdf — censoring rate vs log(median follow-up), coloured by regime
  event_rate_bars.pdf    — sorted event rates across all datasets in the group

LaTeX table
-----------
  dataset_stats.tex / datasets_stats.csv  (group-specific)

Usage
-----
    uv run python -m survpfn.xai.dataset_stats                      # group=all
    uv run python -m survpfn.xai.dataset_stats --group public
    uv run python -m survpfn.xai.dataset_stats --group ehr
    uv run python -m survpfn.xai.dataset_stats --group public+ehr
    uv run python -m survpfn.xai.dataset_stats --group survset
    uv run python -m survpfn.xai.dataset_stats --group ormoni
    uv run python -m survpfn.xai.dataset_stats --group cr
    uv run python -m survpfn.xai.dataset_stats --group all --no-plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Domain / display / citation maps
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_MAP = {
    "VETERANS":                 "Oncology",
    "WHAS500":                  "Cardiology",
    "METABRIC":                 "Oncology",
    "GBSG":                     "Oncology",
    "FLCHAIN":                  "Haematology",
    "SUPPORT2":                 "Critical care",
    "SEER":                     "Oncology",
    "ORMONI_TIRODEI_CV":        "Cardiology",
    "ORMONI_TIRODEI_MI":        "Cardiology",
    "ORMONI_TIRODEI_STROKE":    "Neurovascular",
    "ORMONI_TIRODEI_MORTALITY": "All-cause mortality",
    "EICU_SURV":                "Critical care (EHR)",
    "MIMIC_SURV_B":             "Critical care (EHR)",
    "FRAMINGHAM":               "Cardiology",
    "PBC2":                     "Hepatology",
    "SUPPORT_CR":               "Critical care",
    "SYNTHETIC_CR":             "Synthetic",
    "SS_CANCER":                "Oncology",
    "SS_BREAST":                "Oncology",
    "SS_GBSG2":                 "Oncology",
    "SS_ROTT2":                 "Oncology",
    "SS_COLON":                 "Oncology",
    "SS_PROSTATE":              "Oncology",
    "SS_OVARIAN":               "Oncology",
    "SS_MELANOMA":              "Oncology",
    "SS_E1684":                 "Oncology",
    "SS_PBC":                   "Hepatology",
    "SS_HEPATOCELLULAR":        "Oncology",
    "SS_NWTCO":                 "Oncology",
    "SS_RETINOPATHY":           "Ophthalmology",
    "SS_HEART":                 "Cardiology",
    "SS_VETERAN":               "Oncology",
    "SS_WHAS500":               "Cardiology",
    "SS_MGUS":                  "Haematology",
    "SS_CGD":                   "Immunology",
    "SS_COST":                  "Oncology",
    "SS_LEUKSURV":              "Haematology",
    "SS_DIALYSIS":              "Nephrology",
    "SS_ACTG":                  "Infectious disease",
    "SS_RHC":                   "Critical care",
    "SS_VLBW":                  "Neonatology",
    "SS_GRACE":                 "Cardiology",
    "SS_TRACE":                 "Cardiology",
    "SS_SUPPORT2":              "Critical care",
    "SS_DLBCL":                 "Oncology",
    "SS_DIABETES":              "Endocrinology",
    "SS_FLCHAIN":               "Haematology",
    "SS_FRAMINGHAM":            "Cardiology",
}

# Broad domain groups for colouring
DOMAIN_BROAD = {
    "Oncology":           "Oncology",
    "Cardiology":         "Cardio/Vascular",
    "Neurovascular":      "Cardio/Vascular",
    "All-cause mortality":"Other clinical",
    "Haematology":        "Other clinical",
    "Hepatology":         "Other clinical",
    "Ophthalmology":      "Other clinical",
    "Immunology":         "Other clinical",
    "Nephrology":         "Other clinical",
    "Neonatology":        "Other clinical",
    "Infectious disease": "Other clinical",
    "Endocrinology":      "Other clinical",
    "Critical care":      "Critical care",
    "Critical care (EHR)":"Critical care (EHR)",
    "Synthetic":          "Synthetic",
}

DISPLAY_NAME = {
    "VETERANS":                 "Veterans",
    "WHAS500":                  "WHAS500",
    "METABRIC":                 "METABRIC",
    "GBSG":                     "GBSG",
    "FLCHAIN":                  "FLCHAIN",
    "SUPPORT2":                 "SUPPORT2",
    "SEER":                     "SEER",
    "EICU_SURV":                "eICU",
    "MIMIC_SURV_B":             "MIMIC-IV",
    "ORMONI_TIRODEI_CV":        "OT-CV",
    "ORMONI_TIRODEI_MI":        "OT-MI",
    "ORMONI_TIRODEI_STROKE":    "OT-Stroke",
    "ORMONI_TIRODEI_MORTALITY": "OT-Mortality",
    "FRAMINGHAM":               "Framingham",
    "PBC2":                     "PBC2",
    "SUPPORT_CR":               "SUPPORT (CR)",
    "SYNTHETIC_CR":             "Synthetic (CR)",
}

CITATION_MAP = {
    "VETERANS":     {"cite_key": "prentice1973exponential",
                     "authors":  "Prentice (1973)",
                     "description": "VA RCT: standard vs. test chemotherapy, 137 male lung-cancer patients."},
    "WHAS500":      {"cite_key": "hosmer2008applied",
                     "authors":  "Hosmer et al. (2008)",
                     "description": "Worcester Heart Attack Study; 500 AMI patients (textbook running example)."},
    "METABRIC":     {"cite_key": "curtis2012genomic",
                     "authors":  "Curtis et al. (2012)",
                     "description": "Integrated genomic/transcriptomic analysis of $\\sim$2,000 breast tumours."},
    "GBSG":         {"cite_key": "schumacher1994randomized",
                     "authors":  "Schumacher et al. (1994)",
                     "description": "German Breast Cancer Study Group 2$\\times$2 RCT; 686 node-positive patients."},
    "FLCHAIN":      {"cite_key": "dispenzieri2012use",
                     "authors":  "Dispenzieri et al. (2012)",
                     "description": "Mayo Clinic serum free-light-chain assay; 7,874 community residents."},
    "SUPPORT2":     {"cite_key": "knaus1995support",
                     "authors":  "Knaus et al. (1995)",
                     "description": "SUPPORT: 9,105 seriously ill adults; 180-day survival prognostic model."},
    "SEER":         {"cite_key": "nci2023seer",
                     "authors":  "NCI SEER (2023)",
                     "description": "NCI breast-cancer registry; population-based incidence and survival data."},
    "EICU_SURV":    {"cite_key": "pollard2018eicu",
                     "authors":  "Pollard et al. (2018)",
                     "description": "eICU Collaborative Research Database v2.0; $>$200k ICU admissions, 208 US hospitals."},
    "MIMIC_SURV_B": {"cite_key": "johnson2023mimiciv",
                     "authors":  "Johnson et al. (2023)",
                     "description": "MIMIC-IV v3.1; Beth Israel Deaconess Medical Center 2008--2019; first ICU stay per patient."},
    "FRAMINGHAM":   {"cite_key": "dawber1951epidemiological",
                     "authors":  "Dawber et al. (1951)",
                     "description": "Framingham Heart Study; CVD vs. non-CVD mortality competing risks."},
    "PBC2":         {"cite_key": "murtaugh1994primary",
                     "authors":  "Murtaugh et al. (1994)",
                     "description": "Mayo Clinic PBC longitudinal dataset."},
    "SUPPORT_CR":   {"cite_key": "knaus1995support",
                     "authors":  "Knaus et al. (1995)",
                     "description": "SUPPORT reframed as competing risks (cancer vs. non-cancer death)."},
    "SYNTHETIC_CR": {"cite_key": "", "authors": "Synthetic",
                     "description": "Simulated 2-cause competing-risks dataset for method validation."},
    "SS_CANCER":        {"cite_key": "loprinzi1994prospective",  "authors": "Loprinzi et al. (1994)"},
    "SS_BREAST":        {"cite_key": "schumacher1994randomized", "authors": "Schumacher et al. (1994)"},
    "SS_GBSG2":         {"cite_key": "sauerbrei1999modelling",   "authors": "Sauerbrei et al. (1999)"},
    "SS_ROTT2":         {"cite_key": "royston2013external",      "authors": "Royston \\& Altman (2013)"},
    "SS_COLON":         {"cite_key": "moertel1990levamisole",    "authors": "Moertel et al. (1990)"},
    "SS_PROSTATE":      {"cite_key": "andrews1985data",          "authors": "Andrews \\& Herzberg (1985)"},
    "SS_OVARIAN":       {"cite_key": "edmonson1979different",    "authors": "Edmonson et al. (1979)"},
    "SS_MELANOMA":      {"cite_key": "drzewiecki1980biopsy",     "authors": "Drzewiecki et al. (1980)"},
    "SS_E1684":         {"cite_key": "kirkwood1996interferon",   "authors": "Kirkwood et al. (1996)"},
    "SS_PBC":           {"cite_key": "fleming1991counting",      "authors": "Fleming \\& Harrington (1991)"},
    "SS_HEPATOCELLULAR":{"cite_key": "llovet1999prognosis",      "authors": "Llovet et al. (1999)"},
    "SS_NWTCO":         {"cite_key": "dangelo1989treatment",     "authors": "D'Angio et al. (1989)"},
    "SS_RETINOPATHY":   {"cite_key": "huster1989modelling",      "authors": "Huster et al. (1989)"},
    "SS_HEART":         {"cite_key": "crowley1977covariance",    "authors": "Crowley \\& Hu (1977)"},
    "SS_VETERAN":       {"cite_key": "kalbfleisch1980statistical","authors": "Kalbfleisch \\& Prentice (1980)"},
    "SS_WHAS500":       {"cite_key": "hosmer2008applied",        "authors": "Hosmer et al. (2008)"},
    "SS_MGUS":          {"cite_key": "kyle2002longterm",         "authors": "Kyle et al. (2002)"},
    "SS_CGD":           {"cite_key": "fleming1991counting",      "authors": "Fleming \\& Harrington (1991)"},
    "SS_COST":          {"cite_key": "goldberg2004randomized",   "authors": "Goldberg et al. (2004)"},
    "SS_LEUKSURV":      {"cite_key": "henderson2002modeling",    "authors": "Henderson et al. (2002)"},
    "SS_DIALYSIS":      {"cite_key": "wolfe1999comparison",      "authors": "Wolfe et al. (1999)"},
    "SS_ACTG":          {"cite_key": "hammer1997controlled",     "authors": "Hammer et al. (1997)"},
    "SS_RHC":           {"cite_key": "connors1996effectiveness", "authors": "Connors et al. (1996)"},
    "SS_VLBW":          {"cite_key": "oshea1992prenatal",        "authors": "O'Shea et al. (1992)"},
    "SS_GRACE":         {"cite_key": "fox2006prediction",        "authors": "Fox et al. (2006)"},
    "SS_TRACE":         {"cite_key": "kober1995clinical",        "authors": "K{\\o}ber et al. (1995)"},
    "SS_SUPPORT2":      {"cite_key": "knaus1995support",         "authors": "Knaus et al. (1995)"},
    "SS_DLBCL":         {"cite_key": "rosenwald2002use",         "authors": "Rosenwald et al. (2002)"},
    "SS_DIABETES":      {"cite_key": "collett2015modelling",     "authors": "Collett (2015)"},
    "SS_FLCHAIN":       {"cite_key": "dispenzieri2012use",       "authors": "Dispenzieri et al. (2012)"},
    "SS_FRAMINGHAM":    {"cite_key": "dawber1951epidemiological","authors": "Dawber et al. (1951)"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_regime(n: int, is_cr: bool = False) -> str:
    tag = " (CR)" if is_cr else ""
    if n < 500:
        return f"small{tag}"
    elif n < 5_000:
        return f"medium{tag}"
    else:
        return f"large{tag}"


def _cite(name: str) -> str:
    info = CITATION_MAP.get(name, {})
    key = info.get("cite_key", "")
    return f"\\cite{{{key}}}" if key else ""


def _feature_types(df: pd.DataFrame, dur_col: str, ev_col: str):
    """Return (n_numeric, n_categorical, n_binary)."""
    X = df.drop(columns=[dur_col, ev_col], errors="ignore")
    num = X.select_dtypes(include=[np.number]).columns
    cat = X.select_dtypes(exclude=[np.number]).columns
    binary = [c for c in X.columns if X[c].nunique() <= 2]
    return len(num), len(cat), len(binary)


def _missing_pct(df: pd.DataFrame, dur_col: str, ev_col: str) -> float:
    X = df.drop(columns=[dur_col, ev_col], errors="ignore")
    if X.empty:
        return 0.0
    return float(X.isnull().mean().mean() * 100)


def _mean_abs_corr(df: pd.DataFrame, dur_col: str, ev_col: str,
                   max_feats: int = 60) -> float:
    """Mean |Pearson r| among numeric features (sampled to max_feats)."""
    X = df.drop(columns=[dur_col, ev_col], errors="ignore")
    Xn = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    Xn = Xn.loc[:, Xn.std() > 0]
    if Xn.shape[1] < 2:
        return float("nan")
    if Xn.shape[1] > max_feats:
        Xn = Xn.iloc[:, :max_feats]
    corr = Xn.corr(method="pearson").values
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(np.nanmean(np.abs(corr[mask])))


# ─────────────────────────────────────────────────────────────────────────────
# Stats collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_stats(names: list[str], is_cr: bool = False) -> pd.DataFrame:
    """Load datasets and compute per-dataset statistics."""
    from survpfn.dataloaders import BENCHMARK_DATASETS, CR_DATASETS
    try:
        from survpfn.models.shared.binning import adaptive_num_durations
    except ImportError:
        adaptive_num_durations = None

    registry = CR_DATASETS if is_cr else BENCHMARK_DATASETS
    rows = []

    for name in names:
        loader = registry.get(name)
        if loader is None:
            # try the other registry as fallback
            loader = (CR_DATASETS if not is_cr else BENCHMARK_DATASETS).get(name)
        if loader is None:
            print(f"  [SKIP] {name}: not found in registry")
            continue
        try:
            print(f"  Loading {name} ...")
            df, dur_col, ev_col = loader()
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            continue

        n = len(df)
        p = len(df.columns) - 2
        event_mask = df[ev_col] > 0
        event_rate = float(event_mask.mean())
        censoring_rate = 1.0 - event_rate

        dur = df[dur_col].astype(float)
        median_fu = float(dur.median())
        q25_fu    = float(dur.quantile(0.25))
        q75_fu    = float(dur.quantile(0.75))
        iqr_fu    = q75_fu - q25_fu
        max_fu    = float(dur.max())

        n_num, n_cat, n_bin = _feature_types(df, dur_col, ev_col)
        miss = _missing_pct(df, dur_col, ev_col)
        mac  = _mean_abs_corr(df, dur_col, ev_col)

        n_bins = float("nan")
        if adaptive_num_durations is not None:
            try:
                n_bins = int(adaptive_num_durations(
                    dur.values, event_mask.astype(int).values
                ))
            except Exception:
                pass

        n_causes = int(df[ev_col].max()) if is_cr and df[ev_col].max() > 0 else 1

        domain = DOMAIN_MAP.get(name, "Medicine")
        regime = get_regime(n, is_cr=is_cr)
        dtype  = "CR" if is_cr else "SR"

        rows.append({
            "Dataset":        name,
            "N":              n,
            "Features":       p,
            "Event_rate":     round(event_rate, 4),
            "Censoring_rate": round(censoring_rate, 4),
            "Median_followup": round(median_fu, 1),
            "IQR_followup":   round(iqr_fu, 1),
            "Max_followup":   round(max_fu, 1),
            "N_numeric":      n_num,
            "N_categorical":  n_cat,
            "N_binary":       n_bin,
            "Missing_pct":    round(miss, 2),
            "Mean_abs_corr":  round(mac, 3) if not np.isnan(mac) else float("nan"),
            "N_bins":         n_bins,
            "N_causes":       n_causes,
            "Domain":         domain,
            "Regime":         regime,
            "Type":           dtype,
            "Cite":           _cite(name),
            "Authors":        CITATION_MAP.get(name, {}).get("authors", ""),
        })

    return pd.DataFrame(rows)


def resolve_group(group: str):
    """Return (names, is_cr) from a group name."""
    from survpfn.dataloaders import DATASET_GROUPS, CR_DATASETS
    if group == "cr":
        return DATASET_GROUPS["cr"], True
    names = DATASET_GROUPS.get(group)
    if names is None:
        raise ValueError(f"Unknown group '{group}'.")
    return names, False


# ─────────────────────────────────────────────────────────────────────────────
# Diversity plots
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — stable across groups
_BROAD_PALETTE = {
    "Oncology":            "#e41a1c",
    "Cardio/Vascular":     "#377eb8",
    "Critical care":       "#ff7f00",
    "Critical care (EHR)": "#a65628",
    "Other clinical":      "#4daf4a",
    "Synthetic":           "#999999",
}


def _broad(domain: str) -> str:
    return DOMAIN_BROAD.get(domain, "Other clinical")


def _regime_palette():
    return {"small": "#1b9e77", "medium": "#d95f02", "large": "#7570b3",
            "small (CR)": "#66c2a5", "medium (CR)": "#fc8d62", "large (CR)": "#8da0cb"}


def plot_diversity_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bubble chart: x = log10(N), y = event rate.
    Bubble area ∝ feature count.  Colour = broad domain.
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))

    broad_domains = sorted(df["Domain"].map(_broad).unique())
    for _, row in df.iterrows():
        bd = _broad(row["Domain"])
        color = _BROAD_PALETTE.get(bd, "#888888")
        size  = max(30, row["Features"] * 3.5)
        ax.scatter(np.log10(row["N"]), row["Event_rate"],
                   s=size, color=color, alpha=0.75,
                   edgecolors="white", linewidths=0.6, zorder=3)
        label = DISPLAY_NAME.get(row["Dataset"],
                                  row["Dataset"].replace("SS_", ""))
        ax.annotate(label, (np.log10(row["N"]), row["Event_rate"]),
                    textcoords="offset points", xytext=(0, 5),
                    fontsize=6.5, ha="center", alpha=0.9)

    # Domain legend
    handles = [
        mpatches.Patch(facecolor=_BROAD_PALETTE.get(bd, "#888"),
                       edgecolor="white", label=bd)
        for bd in broad_domains if bd in _BROAD_PALETTE
    ]
    leg1 = ax.legend(handles=handles, title="Domain", fontsize=8,
                     title_fontsize=8, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)

    # Feature-count size legend
    for feats, lab in [(10, "10 feats"), (50, "50 feats"), (150, "150 feats")]:
        ax.scatter([], [], s=max(30, feats * 3.5), color="gray", alpha=0.6, label=lab)
    ax.legend(title="Feature count", fontsize=7, title_fontsize=8,
              loc="lower right", framealpha=0.9)

    ax.set_xlabel(r"Sample size [$\log_{10}$ N]", fontsize=10)
    ax.set_ylabel("Event rate", fontsize=10)
    ax.set_title("Dataset landscape: sample size vs event rate", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_feature_composition(df: pd.DataFrame, out_path: Path) -> None:
    """
    Stacked horizontal bar: continuous / binary / categorical features per dataset.
    Sorted by total feature count descending.
    """
    if df.empty:
        return
    df = df.copy().sort_values("Features", ascending=True)
    labels = [DISPLAY_NAME.get(r, r.replace("SS_", "")) for r in df["Dataset"]]
    n_cont = df["N_numeric"] - df["N_binary"]
    n_bin  = df["N_binary"]
    n_cat  = df["N_categorical"]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.32)))
    y = np.arange(len(df))
    bar_h = 0.65

    ax.barh(y, n_cont, bar_h, color="#4575b4", label="Continuous")
    ax.barh(y, n_bin,  bar_h, left=n_cont, color="#f46d43", label="Binary")
    ax.barh(y, n_cat,  bar_h, left=n_cont + n_bin, color="#74c476", label="Categorical")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(5, 8 - len(df) // 12))
    ax.set_xlabel("Number of features", fontsize=10)
    ax.set_title("Feature composition per dataset", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_censoring_vs_followup(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter: x = censoring rate, y = log10(median follow-up).
    Colour = size regime (small / medium / large).
    """
    if df.empty:
        return
    palette = _regime_palette()
    fig, ax = plt.subplots(figsize=(7, 5))

    regimes = df["Regime"].unique()
    for regime in sorted(regimes):
        sub = df[df["Regime"] == regime]
        color = palette.get(regime, "#888888")
        ax.scatter(sub["Censoring_rate"],
                   np.log10(sub["Median_followup"].clip(lower=0.01)),
                   s=60, color=color, alpha=0.8, edgecolors="white",
                   linewidths=0.6, label=regime, zorder=3)

    for _, row in df.iterrows():
        label = DISPLAY_NAME.get(row["Dataset"],
                                  row["Dataset"].replace("SS_", ""))
        ax.annotate(label,
                    (row["Censoring_rate"],
                     np.log10(max(row["Median_followup"], 0.01))),
                    textcoords="offset points", xytext=(3, 2),
                    fontsize=6, alpha=0.85)

    ax.set_xlabel("Censoring rate", fontsize=10)
    ax.set_ylabel(r"Median follow-up [$\log_{10}$]", fontsize=10)
    ax.set_title("Censoring vs follow-up by size regime", fontsize=11)
    ax.legend(title="Regime", fontsize=8, title_fontsize=8,
              framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_event_rate_bars(df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of event rates, sorted descending, coloured by broad domain.
    """
    if df.empty:
        return
    df = df.copy().sort_values("Event_rate", ascending=True)
    labels = [DISPLAY_NAME.get(r, r.replace("SS_", "")) for r in df["Dataset"]]
    colors = [_BROAD_PALETTE.get(_broad(d), "#888") for d in df["Domain"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.28)))
    y = np.arange(len(df))
    ax.barh(y, df["Event_rate"], 0.7, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(5, 8 - len(df) // 12))
    ax.set_xlabel("Event rate", fontsize=10)
    ax.set_title("Event rate across datasets", fontsize=11)
    ax.axvline(df["Event_rate"].mean(), color="black", linestyle="--",
               linewidth=1, label=f"mean = {df['Event_rate'].mean():.2f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Domain colour legend
    unique_broad = df["Domain"].map(_broad).unique()
    handles = [
        mpatches.Patch(facecolor=_BROAD_PALETTE.get(bd, "#888"),
                       edgecolor="white", label=bd)
        for bd in sorted(unique_broad) if bd in _BROAD_PALETTE
    ]
    ax.legend(handles=handles, title="Domain", fontsize=7, title_fontsize=8,
              loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_missing_overview(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter: x = dataset (sorted by N), y = missing_pct.
    Highlights datasets with any missing data.
    """
    has_missing = df[df["Missing_pct"] > 0]
    if has_missing.empty:
        return
    df_s = df.sort_values("N")
    labels = [DISPLAY_NAME.get(r, r.replace("SS_", "")) for r in df_s["Dataset"]]
    colors = ["#d73027" if m > 0 else "#4575b4" for m in df_s["Missing_pct"]]

    fig, ax = plt.subplots(figsize=(max(6, len(df_s) * 0.35), 4))
    x = np.arange(len(df_s))
    bars = ax.bar(x, df_s["Missing_pct"], color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=max(5, 8 - len(df_s) // 12))
    ax.set_ylabel("Overall missing %", fontsize=10)
    ax.set_title("Missing data by dataset", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


def plot_diversity(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate all diversity figures and save to out_dir."""
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating diversity figures ...")
    plot_diversity_scatter(df,        out_dir / "diversity_scatter.pdf")
    plot_feature_composition(df,      out_dir / "feature_composition.pdf")
    plot_censoring_vs_followup(df,    out_dir / "censoring_vs_followup.pdf")
    plot_event_rate_bars(df,          out_dir / "event_rate_bars.pdf")
    plot_missing_overview(df,         out_dir / "missing_overview.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_row(row: pd.Series, is_cr: bool = False) -> str:
    name  = row["Dataset"]
    disp  = DISPLAY_NAME.get(name, name.replace("SS_", "SS--").replace("_", " "))
    n_fmt = f"{int(row['N']):,}"
    er    = f"{row['Event_rate']:.2f}"
    cr_   = f"{row['Censoring_rate']:.2f}"
    mfu   = f"{row['Median_followup']:.0f}"
    iqr   = f"{row['IQR_followup']:.0f}"
    miss  = f"{row['Missing_pct']:.1f}"
    feat  = str(int(row["Features"]))
    bins  = str(int(row["N_bins"])) if not np.isnan(row["N_bins"]) else "--"
    cite  = row["Cite"]

    if is_cr:
        k = int(row["N_causes"])
        return (f"{disp} & {n_fmt} & {feat} & {er} ($K$={k}) & {cr_} "
                f"& {mfu} ({iqr}) & {miss} & {bins} & {row['Domain']} "
                f"& {row['Regime']} & {cite} \\\\")
    return (f"{disp} & {n_fmt} & {feat} & {er} & {cr_} "
            f"& {mfu} ({iqr}) & {miss} & {bins} & {row['Domain']} "
            f"& {row['Regime']} & {cite} \\\\")


def generate_latex_table(df: pd.DataFrame, out_dir: Path, group: str) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    is_cr = (df["Type"] == "CR").all()
    ncols = 11

    header = (
        r"\textbf{Dataset} & \textbf{$N$} & \textbf{Feat.} "
        r"& \textbf{Event rate} & \textbf{Cens.\,\%} "
        r"& \textbf{Med.\,FU (IQR)} & \textbf{Miss.\,\%} "
        r"& \textbf{Bins} & \textbf{Domain} & \textbf{Regime} "
        r"& \textbf{Source} \\"
    )
    col_spec = r"lrrrrrrrllll"

    latex = []
    latex.append(rf"\begin{{longtable}}{{{col_spec}}}")
    latex.append(
        r"\caption{\textbf{Dataset characteristics.} "
        r"$N$: number of subjects. "
        r"Event rate: proportion of uncensored observations. "
        r"Cens.\,\%: censoring percentage (= 1 $-$ event rate). "
        r"Med.\,FU: median follow-up time in dataset-native units (IQR in parentheses). "
        r"Miss.\,\%: mean fraction of missing values across all feature columns. "
        r"Bins: adaptive time-discretisation intervals for discrete-time heads. "
        r"SurvSet datasets~\cite{drysdale2022survset} are prefixed `SS--'. "
        r"EHR datasets (eICU, MIMIC-IV) use a static 24\,h feature snapshot; "
        r"diagnosis features are time-filtered to prevent discharge-diagnosis leakage."
        rf"}} \label{{tab:datasets_{group}}} \\"
    )
    latex.append(r"\toprule")
    latex.append(header)
    latex.append(r"\midrule")
    latex.append(r"\endfirsthead")
    latex.append(rf"\multicolumn{{{ncols}}}{{c}}{{Table continued}} \\")
    latex.append(r"\toprule")
    latex.append(header)
    latex.append(r"\midrule")
    latex.append(r"\endhead")
    latex.append(r"\midrule")
    latex.append(rf"\multicolumn{{{ncols}}}{{r}}{{\textit{{Continued on next page}}}} \\")
    latex.append(r"\endfoot")
    latex.append(r"\bottomrule")
    latex.append(r"\endlastfoot")

    # Section ordering
    sections: list[tuple[str, list[str]]] = []

    if group in ("public", "public+ehr"):
        sections.append(("(a)~Standard Public Benchmarks",
                          ["VETERANS", "WHAS500", "METABRIC", "GBSG",
                           "FLCHAIN", "SUPPORT2", "SEER"]))
    if group in ("ehr", "public+ehr"):
        sections.append(("(b)~Large-Scale EHR Datasets\\textdagger",
                          ["EICU_SURV", "MIMIC_SURV_B"]))
    if group == "ormoni":
        sections.append(("(c)~Institutional Datasets (OrmoniTirodei)",
                          ["ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI",
                           "ORMONI_TIRODEI_STROKE", "ORMONI_TIRODEI_MORTALITY"]))
    if group == "survset":
        ss = df["Dataset"].tolist()
        sections.append((r"(d)~SurvSet Benchmark Subset~\cite{drysdale2022survset}", ss))
    if group == "cr":
        sections.append(("Competing-Risks Datasets",
                          ["FRAMINGHAM", "PBC2", "SUPPORT_CR", "SYNTHETIC_CR"]))
    if group == "all":
        # Multi-section: public, ehr, survset
        sections = [
            ("(a)~Standard Public Benchmarks",
             ["VETERANS", "WHAS500", "METABRIC", "GBSG", "FLCHAIN", "SUPPORT2", "SEER"]),
            ("(b)~Large-Scale EHR Datasets",
             ["EICU_SURV", "MIMIC_SURV_B"]),
            (r"(c)~SurvSet Benchmark Subset~\cite{drysdale2022survset}",
             df[df["Dataset"].str.startswith("SS_")]["Dataset"].tolist()),
        ]

    if sections:
        for i, (sec_label, sec_names) in enumerate(sections):
            if i > 0:
                latex.append(r"\midrule")
            latex.append(rf"\multicolumn{{{ncols}}}{{c}}{{\textit{{{sec_label}}}}} \\")
            latex.append(r"\midrule")
            for name in sec_names:
                row = df[df["Dataset"] == name]
                if row.empty:
                    continue
                latex.append(_fmt_row(row.iloc[0], is_cr=is_cr))
    else:
        # Flat listing
        for _, row in df.iterrows():
            latex.append(_fmt_row(row, is_cr=is_cr))

    latex.append(r"\end{longtable}")
    tex = "\n".join(latex)

    tex_path = out_dir / "dataset_stats.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table saved to {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from survpfn.dataloaders import DATASET_GROUPS

    parser = argparse.ArgumentParser(
        description="Dataset statistics and diversity figures for SurvPFN."
    )
    parser.add_argument(
        "--group", choices=list(DATASET_GROUPS), default="all",
        help="Dataset group to process (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/xai/datasets"),
        help="Root output directory; group sub-folder is created automatically.",
    )
    parser.add_argument("--no-latex", action="store_true",
                        help="Skip LaTeX table generation.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip diversity figure generation.")
    args = parser.parse_args()

    group   = args.group
    out_dir = args.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset statistics — group: {group}")
    print(f"Output: {out_dir}\n")

    names, is_cr = resolve_group(group)
    df = collect_stats(names, is_cr=is_cr)

    # Always save CSV
    csv_path = out_dir / "dataset_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nStats CSV saved to {csv_path}")
    print(df[["Dataset", "N", "Features", "Event_rate", "Censoring_rate",
               "Median_followup", "Missing_pct"]].to_string(index=False))

    # LaTeX table
    if not args.no_latex:
        generate_latex_table(df, out_dir, group)

    # Diversity figures
    if not args.no_plots:
        plot_diversity(df, out_dir / "figures")

    # Citation JSON (for bibliography tooling)
    cite_path = out_dir / "citations.json"
    with open(cite_path, "w") as f:
        json.dump(CITATION_MAP, f, indent=2)
    print(f"Citations saved to {cite_path}")


if __name__ == "__main__":
    main()
