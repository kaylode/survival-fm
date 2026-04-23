import json
import numpy as np
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

def _load_sig_map(sig_file: str) -> dict[tuple[str, str], str]:
    """Load a significance file and return {(dataset, model): '*'} map."""
    p = Path(sig_file)
    if not p.exists():
        warnings.warn(f"Significance file not found: {sig_file}")
        return {}

    if p.suffix == ".json":
        raw = json.loads(p.read_text())
        result = {}
        for ds, models in raw.items():
            for mdl, marker in models.items():
                result[(ds, mdl)] = marker
        return result
    return {}

def generate_table(sig_file=None, output_file=None):
    base_dir = Path("/home/mpham/workspace/source/ehrfm/survpfn/results/benchmark/ORMONI_TIRODEI_MORTALITY")
    dataset_name = "ORMONI_TIRODEI_MORTALITY"
    
    # Map LaTeX model names to their corresponding folder names
    groups = [
        ("Baselines", [
            ("Cox PH", "cox"),
            ("RSF", "rsf"),
            ("GBSA", "gbsa")
        ]),
        ("Deep Models", [
            ("DeepSurv", "deepsurv"),
            ("MLP-MTLR", "mtlr"),
            ("MLP-DeepHit", "deephit_single"),
            ("DySurv", "dysurv"),
            ("SurvTRACE", "survtrace")
        ]),
        ("Zero-Shot", [
            ("TabPFN-ZS", "tabpfn_zeroshot_perbin_time_ens"),
            ("TabDPT-ZS", "tabdpt_zeroshot_perbin_time_ens"),
            ("TabICL-ZS", "tabicl_zeroshot_perbin_time_ens")
        ]),
        ("Finetune-Surv", [
            ("TabPFN-FT-MTLR", "tabpfn_embedding_mtlr"),
            ("TabDPT-FT-MTLR", "tabdpt_embedding_mtlr"),
            ("TabICL-FT-MTLR", "tabicl_embedding_mtlr")
        ])
    ]
    
    metrics = ["TD-CI_q25", "TD-CI_q50", "TD-CI_q75"]
    
    # Load significance map
    sig_map = {}
    if sig_file:
        sig_map = _load_sig_map(sig_file)
    
    # Dictionary to store results: {model_folder: {metric: [values...]}}
    results = defaultdict(lambda: defaultdict(list))
    
    # Parse metrics.json files
    for metrics_path in base_dir.rglob("metrics.json"):
        parts = metrics_path.parts
        try:
            model_folder = parts[-3]
        except IndexError:
            continue
            
        with open(metrics_path, 'r') as f:
            try:
                data = json.load(f)
                for metric in metrics:
                    if metric in data:
                        results[model_folder][metric].append(data[metric])
            except json.JSONDecodeError:
                pass

    # Calculate means and stds
    stats = {}
    for model_folder, model_metrics in results.items():
        stats[model_folder] = {}
        for metric in metrics:
            vals = model_metrics[metric]
            if vals:
                stats[model_folder][metric] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals)
                }

    # Find best and second-best for each metric among the models listed
    best_vals = {m: [] for m in metrics}
    for _, models in groups:
        for _, folder in models:
            if folder in stats:
                for metric in metrics:
                    if metric in stats[folder]:
                        best_vals[metric].append(stats[folder][metric]["mean"])
    
    thresholds = {}
    for metric in metrics:
        sorted_vals = sorted(best_vals[metric], reverse=True)
        best = sorted_vals[0] if len(sorted_vals) > 0 else -1
        second_best = sorted_vals[1] if len(sorted_vals) > 1 else -1
        thresholds[metric] = (best, second_best)

    # Helper function to format cell
    def format_cell(folder, metric):
        if folder not in stats or metric not in stats[folder]:
            return "—"
        
        mean_val = stats[folder][metric]["mean"]
        std_val = stats[folder][metric]["std"]
        
        mean_str = f"{mean_val:.3f}"
        
        best, second_best = thresholds[metric]
        eps = 1e-6
        if abs(mean_val - best) < eps:
            mean_str = f"\\textbf{{{mean_str}}}"
        elif abs(mean_val - second_best) < eps:
            mean_str = f"\\underline{{{mean_str}}}"
            
        # Significance marker
        sig = sig_map.get((dataset_name, folder), "")
        if sig:
            mean_str = f"{mean_str}^{{\\ast}}"
            
        return f"${mean_str}_{{{{\\pm {std_val:.3f}}}}}$"

    # Generate LaTeX
    lines = []
    _sig_note = (
        " $^{\\ast}$~indicates statistically significant improvement over the "
        "best non-FM baseline ($p<0.05$, Wilcoxon signed-rank test)."
        if sig_file else ""
    )
    
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\textbf{{Time-dependent C-index on the Italian dataset}} (mean~$\\pm$~std, 5-fold CV) evaluated at truncated time horizons corresponding to 25\\%, 50\\%, and 75\\% event-time quantiles. \\textbf{{Bold}} = best, \\underline{{underline}} = second-best.{_sig_note}}}")
    lines.append("\\label{tab:italian:c_td}")
    lines.append("\\resizebox{0.7\\columnwidth}{!}{")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{$C_{\\text{td}}@25\\%$} $\\uparrow$ & \\textbf{$C_{\\text{td}}@50\\%$} $\\uparrow$ & \\textbf{$C_{\\text{td}}@75\\%$} $\\uparrow$ \\\\")
    
    for i, (group_name, models) in enumerate(groups):
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{\\footnotesize {group_name}}}}} \\\\")
            
        for display_name, folder_name in models:
            c25 = format_cell(folder_name, "TD-CI_q25")
            c50 = format_cell(folder_name, "TD-CI_q50")
            c75 = format_cell(folder_name, "TD-CI_q75")
            
            lines.append(f"{display_name} & {c25} & {c50} & {c75} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)
    print(latex_str)
    
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex_str)
        print(f"\nTable successfully saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig-file", type=str, default=None, help="Path to significance_map.json")
    parser.add_argument("--output", type=str, default=None, help="Path to save the .tex file")
    args = parser.parse_args()
    generate_table(sig_file=args.sig_file, output_file=args.output)
