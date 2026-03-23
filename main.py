import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from survpfn.data_loader import load_and_merge_data
from survpfn.preprocessing import (
    clean_and_impute,
    prepare_cox_data,
    prepare_cardiovascular_data,
    prepare_mi_data,
    prepare_stroke_data
)
from survpfn.cox_models import (
    run_univariate_cox,
    run_multivariate_cox,
    run_competing_risks_cox,
    run_kaplan_meier
)
from survpfn.plotting import (
    plot_correlation_matrix,
    plot_feature_importance,
    plot_forest,
    plot_competing_risks_comparison,
    plot_cif
)
from survpfn.deep_surv import train_deepsurv
from survpfn.deep_hit import train_deephit_competing_risks
from survpfn.tree_models import train_rsf, train_gbsa
from survpfn.custom_model import train_custom_torchsurv
from survpfn.embedding_cox import train_embedding_cox
from survpfn.metrics import evaluate_survival_model, run_statistical_tests

def main():
    parser = argparse.ArgumentParser(description="Survival Analysis Benchmark Suite")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning with Optuna")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials per model")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data...")
    df_main_raw = load_and_merge_data("Dataset Sirbu")
    
    # 2. Preprocess Data
    print("Preprocessing data...")
    df_main = clean_and_impute(df_main_raw)
    print(f"Final dataset shape: {df_main.shape}")
    
    results = []    

    tasks = [
        {"name": "Total Mortality", "prep_func": prepare_cox_data, "duration": "Follow Up Data", "event": "Total mortality", "event_val": 1},
        {"name": "CV Mortality", "prep_func": prepare_cardiovascular_data, "duration": "Follow Up Data", "event": "death", "event_val": 1},
        {"name": "Myocardial Infarction", "prep_func": prepare_mi_data, "duration": "MI_date", "event": "events", "event_val": 1},
        {"name": "Stroke (Ictus)", "prep_func": prepare_stroke_data, "duration": "Ictus_date", "event": "events", "event_val": 1}
    ]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    for task in tasks:
        print(f"\n{'='*30}\nTASK: {task['name']}\n{'='*30}")
        df_task = task['prep_func'](df_main)
        
        # Determine numerical columns for scaling
        num_cols = df_task.select_dtypes(include=[np.number]).columns.tolist()
        if task['duration'] in num_cols: num_cols.remove(task['duration'])
        if task['event'] in num_cols: num_cols.remove(task['event'])

        for fold, (train_idx, test_idx) in enumerate(skf.split(df_task, df_task[task['event']])):
            print(f"\n--- Fold {fold+1}/{args.folds} ---")
            df_train_raw = df_task.iloc[train_idx].reset_index(drop=True)
            df_test_raw = df_task.iloc[test_idx].reset_index(drop=True)

            # Independent Scaling
            scaler = StandardScaler()
            df_train = df_train_raw.copy()
            df_test = df_test_raw.copy()
            df_train[num_cols] = scaler.fit_transform(df_train_raw[num_cols])
            df_test[num_cols] = scaler.transform(df_test_raw[num_cols])

            # 1. Kaplan-Meier (Baseline)
            print("Model: Kaplan-Meier (Population Baseline)")
            km_time, km_surv, _ = run_kaplan_meier(df_test, task['duration'], task['event'])
            # KM has no risk scores per individual (everyone gets the same population curve), 
            # so concordance is 0.5 by definition.
            # We simulate a "mean" prediction for individual metrics
            km_risk = np.zeros(len(df_test))
            # Reshape km_surv to (1, n_times) then tile to (n_samples, n_times)
            km_surv_expanded = np.tile(km_surv, (len(df_test), 1))
            metrics_km = evaluate_survival_model(df_train, df_test, task['duration'], task['event'], km_risk, surv_probs=km_surv_expanded, surv_times=km_time)
            results.append({"Task": task['name'], "Model": "Kaplan-Meier", "Fold": fold+1, **metrics_km})

            # 2. Multivariate Cox
            print("Model: Multivariate Cox")
            _, _, _, c_idx = run_multivariate_cox(df_train, df_test, task['duration'], task['event'])
            results.append({"Task": task['name'], "Model": "Multivariate Cox", "Fold": fold+1, "C-index": c_idx})

            # 3. DeepSurv
            print("Model: DeepSurv")
            _, risk_ds, surv_p_ds, surv_t_ds = train_deepsurv(df_train, df_test, task['duration'], task['event'], tune=args.tune, n_trials=args.trials)
            metrics_ds = evaluate_survival_model(df_train, df_test, task['duration'], task['event'], risk_ds, surv_probs=surv_p_ds, surv_times=surv_t_ds)
            results.append({"Task": task['name'], "Model": "DeepSurv", "Fold": fold+1, **metrics_ds})

            # 4. DeepHit (if competing risks or single)
            if task['name'] in ["CV Mortality", "Myocardial Infarction", "Stroke (Ictus)"]:
                print("Model: DeepHit")
                _, surv_df_dh, ev_dh = train_deephit_competing_risks(df_train, df_test, task['duration'], task['event'], tune=args.tune, n_trials=args.trials)
                results.append({"Task": task['name'], "Model": "DeepHit", "Fold": fold+1, "C-index": ev_dh.concordance_td()})

            # 5. Custom TorchSurv Model
            print("Model: Custom TorchSurv")
            _, risk_ts, surv_p_ts, surv_t_ts = train_custom_torchsurv(df_train, df_test, task['duration'], task['event'], tune=args.tune, n_trials=args.trials)
            metrics_ts = evaluate_survival_model(df_train, df_test, task['duration'], task['event'], risk_ts, surv_probs=surv_p_ts, surv_times=surv_t_ts)
            results.append({"Task": task['name'], "Model": "TorchSurv (Custom)", "Fold": fold+1, **metrics_ts})

            # 6. Random Survival Forest
            print("Model: RSF")
            _, risk_rsf, surv_p, surv_t = train_rsf(df_train, df_test, task['duration'], task['event'], tune=args.tune, n_trials=args.trials)
            metrics = evaluate_survival_model(df_train, df_test, task['duration'], task['event'], risk_rsf, surv_probs=surv_p, surv_times=surv_t)
            results.append({"Task": task['name'], "Model": "RSF", "Fold": fold+1, **metrics})

            # 7. Gradient Boosting
            print("Model: GBSA")
            _, risk_gb, surv_p, surv_t = train_gbsa(df_train, df_test, task['duration'], task['event'], tune=args.tune, n_trials=args.trials)
            metrics = evaluate_survival_model(df_train, df_test, task['duration'], task['event'], risk_gb, surv_probs=surv_p, surv_times=surv_t)
            results.append({"Task": task['name'], "Model": "GBSA", "Fold": fold+1, **metrics})

            # 8. Embedding Cox
            print("Model: Embedding Cox")
            _, risk_ec, surv_p_ec, surv_t_ec = train_embedding_cox(df_train, df_test, task['duration'], task['event'], tune=args.tune, n_trials=args.trials)
            metrics_ec = evaluate_survival_model(df_train, df_test, task['duration'], task['event'], risk_ec, surv_probs=surv_p_ec, surv_times=surv_t_ec)
            results.append({"Task": task['name'], "Model": "Embedding Cox", "Fold": fold+1, **metrics_ec})



    # Saving Results
    print("\n--- Saving Results ---")
    df_res = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/model_comparison.csv", index=False)
    
    # Statistical Tests
    print("\n--- Statistical Tests (Wilcoxon against Multivariate Cox) ---")
    stats_df = run_statistical_tests(df_res)
    stats_df.to_csv("results/statistical_pvalues.csv", index=False)
    print(stats_df)

    # Visualization - C-index
    if "C-index" in df_res.columns:
        fig, ax = plt.subplots(figsize=(14, 7))
        df_cindex = df_res.dropna(subset=["C-index"])
        if not df_cindex.empty:
            sns.barplot(data=df_cindex, x="Task", y="C-index", hue="Model",
                        capsize=.1, errorbar="sd", ax=ax)
            ax.set_title("Model Comparison across Tasks (C-index)")
            ax.set_ylim(0.4, 1.0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
        fig.savefig("results/model_comparison_cindex.pdf")
        plt.close(fig)

    # Visualization - AUROC
    if "AUC_mean" in df_res.columns:
        plt.figure(figsize=(14, 7))
        # Filter out models where AUC is NaN
        df_auc = df_res.dropna(subset=["AUC_mean"])
        if not df_auc.empty:
            sns.barplot(data=df_auc, x="Task", y="AUC_mean", hue="Model", capsize=.1, errorbar="sd")
            plt.title("Model Comparison across Tasks (Mean Time-Dependent AUROC)")
            plt.ylim(0.4, 1.0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig("results/model_comparison_auroc.pdf")
            plt.close()

    print("Optimization complete. Final artifacts saved in results/ directory.")
    
if __name__ == "__main__":
    main()
