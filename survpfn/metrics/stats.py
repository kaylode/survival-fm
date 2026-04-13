import numpy as np

def run_statistical_tests(results_df, metric="C-index", reference_model="Multivariate Cox"):
    """
    Run Wilcoxon signed-rank test comparing reference_model to all other models.
    """
    stats_res = []

    for task in results_df["Task"].unique():
        task_df = results_df[results_df["Task"] == task]
        ref_df = task_df[task_df["Model"] == reference_model]
        if ref_df.empty:
            continue

        ref_scores = ref_df.sort_values("Fold")[metric].values

        for model in task_df["Model"].unique():
            if model == reference_model:
                continue

            mod_df = task_df[task_df["Model"] == model].sort_values("Fold")
            mod_scores = mod_df[metric].values

            if len(mod_scores) == len(ref_scores) and len(ref_scores) > 1:
                # Differences
                diff = mod_scores - ref_scores
                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(mod_scores, ref_scores, zero_method='zsplit')
                    except Exception:
                        p_val = np.nan

                stats_res.append({
                    "Task": task,
                    "Reference": reference_model,
                    "Model": model,
                    "Mean_Diff": float(np.mean(diff)),
                    "Median_Diff": float(np.median(diff)),
                    "p-value": p_val,
                    "Significant": bool(p_val < 0.05) if not np.isnan(p_val) else False
                })

    return pd.DataFrame(stats_res)
