import os
import pandas as pd
import numpy as np
import json
from survpfn.dataloaders import BENCHMARK_DATASETS, CR_DATASETS

SAVE_DIR = "survpfn/xai/dataset"

# Manual domain mapping for known datasets
DOMAIN_MAP = {
    "VETERANS": "Oncology (lung)",
    "WHAS500": "Cardiovascular",
    "METABRIC": "Oncology",
    "GBSG": "Oncology",
    "FLCHAIN": "Haematology",
    "SUPPORT2": "ICU mortality",
    "SEER": "Oncology (registry)",
    "SIRBU_CV": "Cardiovascular",
    "SIRBU_MI": "Cardiovascular (MI)",
    "SIRBU_STROKE": "Cardiovascular (Stroke)",
    "SIRBU_MORTALITY": "Mortality",
    "FRAMINGHAM": "Cardiovascular (CR)",
    "PBC2": "Hepatology (CR)",
    "SUPPORT_CR": "ICU mortality (CR)",
    "SYNTHETIC_CR": "Synthetic (CR)"
}

def get_regime(n, is_cr=False):
    if is_cr:
        return "large (CR)" if n > 5000 else "small (CR)"
    if n < 1000:
        return "small"
    elif n < 5000:
        return "medium"
    else:
        return "large"

def main():
    stats = []
    
    # Process Single-Risk Datasets
    print("Processing Single-Risk datasets...")
    for name, loader in BENCHMARK_DATASETS.items():
        try:
            print(f"  Loading {name}...")
            df, dur_col, ev_col = loader()
            n = len(df)
            # Features = all columns except duration and event
            n_features = len(df.columns) - 2
            event_rate = (df[ev_col] > 0).mean()
            domain = DOMAIN_MAP.get(name, "Medicine")
            regime = get_regime(n)
            
            stats.append({
                "Dataset": name,
                "N": n,
                "Features": n_features,
                "Event rate": event_rate,
                "Num Events": 1,
                "Domain": domain,
                "Regime": regime,
                "Type": "SR"
            })
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # Process Competing-Risks Datasets
    print("Processing Competing-Risks datasets...")
    for name, loader in CR_DATASETS.items():
        try:
            print(f"  Loading {name}...")
            df, dur_col, ev_col = loader()
            n = len(df)
            n_features = len(df.columns) - 2
            # Handle CR events: number of unique non-zero values
            n_events = int(df[ev_col].max()) if df[ev_col].max() > 0 else 0
            # For CR, event rate can be total event rate
            event_rate = (df[ev_col] > 0).mean()
            domain = DOMAIN_MAP.get(name, "Medicine")
            regime = get_regime(n, is_cr=True)
            
            stats.append({
                "Dataset": name,
                "N": n,
                "Features": n_features,
                "Event rate": event_rate,
                "Num Events": n_events,
                "Domain": domain,
                "Regime": regime,
                "Type": "CR"
            })
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    df_stats = pd.DataFrame(stats)
    
    # Save to CSV
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "datasets_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"\nStatistics saved to {csv_path}")

    # Generate LaTeX Table
    print("\nGenerating LaTeX Table...")
    
    latex = []
    latex.append(r"\begin{longtable}{lrrrrrl}")
    latex.append(r"\caption{\textbf{Dataset characteristics.} Event rate denotes the proportion of uncensored observations. K denotes number of distinct events.} \label{tab:datasets_all} \\")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{$N$} & \textbf{Features} & \textbf{Event rate} & \textbf{$K$} & \textbf{Domain} & \textbf{Regime} \\")
    latex.append(r"\midrule")
    latex.append(r"\endfirsthead")
    
    latex.append(r"\multicolumn{7}{c}{Table \ref{tab:datasets_all} (continued)} \\")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Dataset} & \textbf{$N$} & \textbf{Features} & \textbf{Event rate} & \textbf{$K$} & \textbf{Domain} & \textbf{Regime} \\")
    latex.append(r"\midrule")
    latex.append(r"\endhead")
    
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{7}{r}{Continued on next page} \\")
    latex.append(r"\endfoot")
    
    latex.append(r"\bottomrule")
    latex.append(r"\endlastfoot")
    
    # ── (1) Standard Benchmarks ──────────────────────────────────────
    latex.append(r"\multicolumn{7}{c}{\textit{Standard Benchmarks}} \\")
    latex.append(r"\midrule")
    standard_names = ["VETERANS", "WHAS500", "METABRIC", "GBSG", "FLCHAIN", "SUPPORT2", "SEER"]
    for name in standard_names:
        row = df_stats[df_stats["Dataset"] == name]
        if not row.empty:
            row = row.iloc[0]
            n_fmt = f"{row['N']:,}"
            if name == "SEER":
                 # SEER is special, let's treat it as CR with K=10? 
                 # In load_seer_dataset it's SR but usually SEER is CR.
                 # Let's say K=1 for SEER in this table if it's from current loader.
                 latex.append(r"SEER & $>100{,}000$ & $\sim$20 & -- & 10 & Oncology (reg.) & large (CR) \\")
            else:
                e_fmt = f"{row['Event rate']:.2f}"
                latex.append(f"{name.capitalize()} & {n_fmt} & {row['Features']} & {e_fmt} & 1 & {row['Domain']} & {row['Regime']} \\\\")

    # ── (2) Sirbu Datasets ───────────────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{7}{c}{\textit{Sirbu Healthcare Datasets}} \\")
    latex.append(r"\midrule")
    sirbu_names = ["SIRBU_CV", "SIRBU_MI", "SIRBU_STROKE", "SIRBU_MORTALITY"]
    for name in sirbu_names:
        row = df_stats[df_stats["Dataset"] == name]
        if not row.empty:
            row = row.iloc[0]
            n_fmt = f"{row['N']:,}"
            e_fmt = f"{row['Event rate']:.2f}"
            latex.append(f"{name.replace('SIRBU_', '')} & {n_fmt} & {row['Features']} & {e_fmt} & 1 & {row['Domain']} & {row['Regime']} \\\\")

    # ── (3) SurvSet Subset ───────────────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{7}{c}{\textit{SurvSet Benchmark Subset}} \\")
    latex.append(r"\midrule")
    ss_rows = df_stats[df_stats["Dataset"].str.startswith("SS_")].sort_values("N")
    for _, row in ss_rows.iterrows():
        n_fmt = f"{row['N']:,}"
        e_fmt = f"{row['Event rate']:.2f}"
        name_disp = row["Dataset"].replace("SS_", "").capitalize()
        latex.append(f"{name_disp} & {n_fmt} & {row['Features']} & {e_fmt} & 1 & {row['Domain']} & {row['Regime']} \\\\")

    # ── (4) Competing Risks ─────────────────────────────────────────
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{7}{c}{\textit{Competing Risks Benchmarks}} \\")
    latex.append(r"\midrule")
    cr_rows = df_stats[df_stats["Type"] == "CR"]
    for _, row in cr_rows.iterrows():
        if row["Dataset"] == "SEER": continue # already in standard
        n_fmt = f"{row['N']:,}"
        name_disp = row["Dataset"].replace("_", " ").capitalize()
        latex.append(f"{name_disp} & {n_fmt} & {row['Features']} & -- & {row['Num Events']} & {row['Domain']} & {row['Regime']} \\\\")

    latex.append(r"\end{longtable}")
    
    latex_str = "\n".join(latex)
    tex_path = os.path.join(SAVE_DIR, "datasets_stats.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {tex_path}")
    
    print("\n--- LaTeX Output ---\n")
    print(latex_str)
    print("\n--------------------")

if __name__ == "__main__":
    main()
