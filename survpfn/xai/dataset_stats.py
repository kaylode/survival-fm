import os
import pandas as pd
import numpy as np
import json
from survpfn.dataloaders import BENCHMARK_DATASETS, CR_DATASETS
from survpfn.models.shared.binning import adaptive_num_durations

SAVE_DIR = "results/xai/datasets"

# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------
DOMAIN_MAP = {
    # Standard benchmarks
    "VETERANS":                 "Oncology (lung)",
    "WHAS500":                  "Cardiology",
    "METABRIC":                 "Oncology (breast)",
    "GBSG":                     "Oncology (breast)",
    "FLCHAIN":                  "Haematology",
    "SUPPORT2":                 "Critical care",
    "SEER":                     "Oncology (breast)",
    # Private / institutional
    "ORMONI_TIRODEI_CV":        "Cardiovascular",
    "ORMONI_TIRODEI_MI":        "Cardiovascular",
    "ORMONI_TIRODEI_STROKE":    "Neurovascular",
    "ORMONI_TIRODEI_MORTALITY": "All-cause mortality",
    # EHR
    "EICU_SURV":                "Critical care",
    "MIMIC_SURV_B":             "Critical care",
    # Competing-risks
    "FRAMINGHAM":               "Cardiology",
    "PBC2":                     "Hepatology",
    "SUPPORT_CR":               "Critical care",
    "SYNTHETIC_CR":             "Synthetic",
    # SurvSet
    "SS_CANCER":                "Oncology (lung)",
    "SS_BREAST":                "Oncology (breast)",
    "SS_GBSG2":                 "Oncology (breast)",
    "SS_ROTT2":                 "Oncology (breast)",
    "SS_COLON":                 "Oncology (colon)",
    "SS_PROSTATE":              "Oncology (prostate)",
    "SS_OVARIAN":               "Oncology (ovarian)",
    "SS_MELANOMA":              "Oncology (melanoma)",
    "SS_E1684":                 "Oncology (melanoma)",
    "SS_PBC":                   "Hepatology",
    "SS_HEPATOCELLULAR":        "Oncology (liver)",
    "SS_NWTCO":                 "Paediatric oncology",
    "SS_RETINOPATHY":           "Ophthalmology",
    "SS_HEART":                 "Cardiology",
    "SS_VETERAN":               "Oncology (lung)",
    "SS_WHAS500":               "Cardiology",
    "SS_MGUS":                  "Haematology",
    "SS_CGD":                   "Immunology",
    "SS_COST":                  "Oncology (colorectal)",
    "SS_LEUKSURV":              "Haematology",
    "SS_DIALYSIS":              "Nephrology",
    "SS_ACTG":                  "Infectious disease",
    "SS_RHC":                   "Critical care",
    "SS_VLBW":                  "Neonatology",
    "SS_GRACE":                 "Cardiology",
    "SS_TRACE":                 "Cardiology",
    "SS_SUPPORT2":              "Critical care",
    "SS_DLBCL":                 "Oncology (lymphoma)",
    "SS_DIABETES":              "Endocrinology",
    "SS_FLCHAIN":               "Haematology",
    "SS_FRAMINGHAM":            "Cardiology",
}

# ---------------------------------------------------------------------------
# Citation mapping  (cite_key → BibTeX key for \cite{} in LaTeX)
# ---------------------------------------------------------------------------
CITATION_MAP = {
    # Standard benchmarks
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
    # EHR
    "EICU_SURV":    {"cite_key": "pollard2018eicu",
                     "authors":  "Pollard et al. (2018)",
                     "description": "eICU CRD: $>$200k ICU admissions, 208 US hospitals (Philips/MIT)."},
    "MIMIC_SURV_B": {"cite_key": "johnson2023mimiciv",
                     "authors":  "Johnson et al. (2023)",
                     "description": "MIMIC-IV: de-identified EHR, Beth Israel Deaconess MC 2008--2019."},
    # Competing risks
    "FRAMINGHAM":   {"cite_key": "dawber1951epidemiological",
                     "authors":  "Dawber et al. (1951)",
                     "description": "Framingham Heart Study; CVD vs. non-CVD mortality competing risks."},
    "PBC2":         {"cite_key": "murtaugh1994primary",
                     "authors":  "Murtaugh et al. (1994)",
                     "description": "Mayo Clinic PBC longitudinal dataset; repeated measurements for joint modelling."},
    "SUPPORT_CR":   {"cite_key": "knaus1995support",
                     "authors":  "Knaus et al. (1995)",
                     "description": "SUPPORT reframed as competing risks (cancer vs. non-cancer death)."},
    "SYNTHETIC_CR": {"cite_key": "",
                     "authors":  "Synthetic",
                     "description": "Simulated 2-cause competing-risks dataset for method validation."},
    # SurvSet
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

# Nice display names for the LaTeX table
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


def get_regime(n, is_cr=False):
    tag = " (CR)" if is_cr else ""
    if n < 500:
        return f"small{tag}"
    elif n < 5000:
        return f"medium{tag}"
    else:
        return f"large{tag}"


def _cite(name: str) -> str:
    """Return a \\cite{key} string, or empty string if no key."""
    info = CITATION_MAP.get(name, {})
    key = info.get("cite_key", "")
    return f"\\cite{{{key}}}" if key else ""


def main():
    stats = []

    # ── Single-Risk Datasets ──────────────────────────────────────────────────
    print("Processing Single-Risk datasets...")
    for name, loader in BENCHMARK_DATASETS.items():
        try:
            print(f"  Loading {name}...")
            df, dur_col, ev_col = loader()
            n = len(df)
            n_features = len(df.columns) - 2
            event_rate = (df[ev_col] > 0).mean()
            domain = DOMAIN_MAP.get(name, "Medicine")
            regime = get_regime(n)
            n_bins = adaptive_num_durations(df[dur_col].values, df[ev_col].values)
            cite = _cite(name)
            authors = CITATION_MAP.get(name, {}).get("authors", "")

            stats.append({
                "Dataset":    name,
                "N":          n,
                "Features":   n_features,
                "Event rate": event_rate,
                "Num Events": 1,
                "Bins":       n_bins,
                "Domain":     domain,
                "Regime":     regime,
                "Type":       "SR",
                "Cite":       cite,
                "Authors":    authors,
            })
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # ── Competing-Risks Datasets ─────────────────────────────────────────────
    print("Processing Competing-Risks datasets...")
    for name, loader in CR_DATASETS.items():
        try:
            print(f"  Loading {name}...")
            df, dur_col, ev_col = loader()
            n = len(df)
            n_features = len(df.columns) - 2
            n_events = int(df[ev_col].max()) if df[ev_col].max() > 0 else 0
            event_rate = (df[ev_col] > 0).mean()
            domain = DOMAIN_MAP.get(name, "Medicine")
            regime = get_regime(n, is_cr=True)
            n_bins = adaptive_num_durations(df[dur_col].values, (df[ev_col] > 0).astype(int).values)
            cite = _cite(name)
            authors = CITATION_MAP.get(name, {}).get("authors", "")

            stats.append({
                "Dataset":    name,
                "N":          n,
                "Features":   n_features,
                "Event rate": event_rate,
                "Num Events": n_events,
                "Bins":       n_bins,
                "Domain":     domain,
                "Regime":     regime,
                "Type":       "CR",
                "Cite":       cite,
                "Authors":    authors,
            })
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    df_stats = pd.DataFrame(stats)

    # Save CSV
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "datasets_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"\nStatistics saved to {csv_path}")

    # ── Generate LaTeX Table ──────────────────────────────────────────────────
    print("\nGenerating LaTeX Table...")

    def fmt_row_sr(name, row):
        """Format a single-risk row."""
        disp = DISPLAY_NAME.get(name, name.replace("SS_", "").replace("_", " ").capitalize())
        n_fmt = f"{int(row['N']):,}"
        e_fmt = f"{row['Event rate']:.2f}"
        cite  = row["Cite"]
        return (
            f"{disp} & {n_fmt} & {int(row['Features'])} & {e_fmt} "
            f"& {int(row['Bins'])} & {row['Domain']} & {row['Regime']} & {cite} \\\\"
        )

    def fmt_row_cr(name, row):
        """Format a competing-risks row (event rate shown as '-' for clarity)."""
        disp = DISPLAY_NAME.get(name, name.replace("_", " ").capitalize())
        n_fmt = f"{int(row['N']):,}"
        e_fmt = f"{row['Event rate']:.2f}"
        k     = int(row["Num Events"])
        cite  = row["Cite"]
        return (
            f"{disp} & {n_fmt} & {int(row['Features'])} & {e_fmt} ($K$={k}) "
            f"& {int(row['Bins'])} & {row['Domain']} & {row['Regime']} & {cite} \\\\"
        )

    latex = []
    latex.append(r"\begin{longtable}{lrrrrlll}")
    latex.append(
        r"\caption{\textbf{Dataset characteristics.} "
        r"$N$: number of subjects. "
        r"Event rate: proportion of uncensored observations; for competing-risks datasets the "
        r"total event rate is shown with the number of distinct causes $K$ in parentheses. "
        r"Bins: number of time-discretisation intervals for discrete-time survival heads "
        r"(computed adaptively from the event distribution). "
        r"Datasets from the SurvSet benchmark~\cite{drysdale2022survset} are prefixed `SS--'."
        r"} \label{tab:datasets_all} \\"
    )
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Dataset} & \textbf{$N$} & \textbf{Feat.} & \textbf{Event rate} "
        r"& \textbf{Bins} & \textbf{Domain} & \textbf{Regime} & \textbf{Source} \\"
    )
    latex.append(r"\midrule")
    latex.append(r"\endfirsthead")

    latex.append(r"\multicolumn{8}{c}{Table~\ref{tab:datasets_all} (continued)} \\")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Dataset} & \textbf{$N$} & \textbf{Feat.} & \textbf{Event rate} "
        r"& \textbf{Bins} & \textbf{Domain} & \textbf{Regime} & \textbf{Source} \\"
    )
    latex.append(r"\midrule")
    latex.append(r"\endhead")

    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{r}{\textit{Continued on next page}} \\")
    latex.append(r"\endfoot")

    latex.append(r"\bottomrule")
    latex.append(r"\endlastfoot")

    # ── (1) Standard Benchmarks ───────────────────────────────────────────────
    latex.append(r"\multicolumn{8}{c}{\textit{(a)~Standard Public Benchmarks}} \\")
    latex.append(r"\midrule")
    standard_names = [
        "VETERANS", "WHAS500", "METABRIC", "GBSG",
        "FLCHAIN", "SUPPORT2", "SEER",
    ]
    for name in standard_names:
        row = df_stats[df_stats["Dataset"] == name]
        if not row.empty:
            latex.append(fmt_row_sr(name, row.iloc[0]))
        else:
            print(f"  [WARN] {name} not found in stats (load may have failed)")

    # ── (2) Large-Scale EHR Datasets ─────────────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{c}{\textit{(b)~Large-Scale EHR Datasets}} \\")
    latex.append(r"\midrule")
    ehr_names = ["EICU_SURV", "MIMIC_SURV_B"]
    for name in ehr_names:
        row = df_stats[df_stats["Dataset"] == name]
        if not row.empty:
            latex.append(fmt_row_sr(name, row.iloc[0]))
        else:
            print(f"  [WARN] {name} not found in stats (load may have failed)")

    # ── (3) Institutional / Private Datasets ─────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{c}{\textit{(c)~Institutional Datasets (OrmoniTirodei)}} \\")
    latex.append(r"\midrule")
    ot_names = [
        "ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI",
        "ORMONI_TIRODEI_STROKE", "ORMONI_TIRODEI_MORTALITY",
    ]
    for name in ot_names:
        row = df_stats[df_stats["Dataset"] == name]
        if not row.empty:
            disp = DISPLAY_NAME.get(name, name)
            r = row.iloc[0]
            n_fmt = f"{int(r['N']):,}"
            e_fmt = f"{r['Event rate']:.2f}"
            latex.append(
                f"{disp} & {n_fmt} & {int(r['Features'])} & {e_fmt} "
                f"& {int(r['Bins'])} & {r['Domain']} & {r['Regime']} & -- \\\\"
            )
        else:
            print(f"  [WARN] {name} not found in stats (load may have failed)")

    # ── (4) SurvSet Subset ────────────────────────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{c}{\textit{(d)~SurvSet Benchmark Subset~\cite{drysdale2022survset}}} \\")
    latex.append(r"\midrule")
    ss_rows = df_stats[df_stats["Dataset"].str.startswith("SS_")].sort_values("N")
    for _, row in ss_rows.iterrows():
        name = row["Dataset"]
        raw  = name.replace("SS_", "")
        disp = "SS--" + raw.capitalize()
        n_fmt = f"{int(row['N']):,}"
        e_fmt = f"{row['Event rate']:.2f}"
        cite  = row["Cite"]
        latex.append(
            f"{disp} & {n_fmt} & {int(row['Features'])} & {e_fmt} "
            f"& {int(row['Bins'])} & {row['Domain']} & {row['Regime']} & {cite} \\\\"
        )

    # ── (5) Competing-Risks Datasets ──────────────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{c}{\textit{(e)~Competing-Risks Datasets}} \\")
    latex.append(r"\midrule")
    cr_rows = df_stats[df_stats["Type"] == "CR"]
    for _, row in cr_rows.iterrows():
        latex.append(fmt_row_cr(row["Dataset"], row))

    latex.append(r"\end{longtable}")

    latex_str = "\n".join(latex)
    tex_path = os.path.join(SAVE_DIR, "datasets_stats.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {tex_path}")

    print("\n--- LaTeX Output ---\n")
    print(latex_str)
    print("\n--------------------")

    # ── Also dump citation map to JSON for bibliography tooling ──────────────
    cite_path = os.path.join(SAVE_DIR, "citations.json")
    with open(cite_path, "w") as f:
        json.dump(CITATION_MAP, f, indent=2)
    print(f"Citation map saved to {cite_path}")


if __name__ == "__main__":
    main()
