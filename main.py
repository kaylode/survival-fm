import os
from src.data_loader import load_and_merge_data
from src.preprocessing import (
    clean_and_impute,
    prepare_cox_data,
    prepare_cardiovascular_data,
    prepare_mi_data,
    prepare_stroke_data
)
from src.cox_models import (
    run_univariate_cox,
    run_multivariate_cox,
    run_competing_risks_cox
)
from src.plotting import (
    plot_correlation_matrix,
    plot_feature_importance,
    plot_forest,
    plot_competing_risks_comparison,
    plot_cif
)
from src.deep_surv import train_deepsurv
from src.deep_hit import train_deephit_competing_risks
from src.tabpfn_modelling import get_tabpfn_embeddings
from src.other_datasets import load_urrah_dataset, load_mimic_dataset

def main():
    # 1. Load Data
    print("Loading data...")
    df_main_raw = load_and_merge_data("Dataset Sirbu")
    
    # 2. Preprocess Data
    print("Preprocessing data...")
    df_main = clean_and_impute(df_main_raw)
    
    print(f"Final dataset shape: {df_main.shape}")
    
    # Optional: Plot correlation
    # plot_correlation_matrix(df_main)

    # ---------------------------
    # General Mortality Model
    # ---------------------------
    print("\n--- General Mortality ---")
    df_mortality = prepare_cox_data(df_main)
    
    print("Running Univariate Cox...")
    univariate_sig, univariate_top20 = run_univariate_cox(df_mortality, "Follow Up Data", "Total mortality")
    plot_feature_importance(univariate_top20, title="Univariate Cox Feature Importance")
    plot_forest(univariate_sig, title="Univariate Cox Forest Plot")

    print("Running Multivariate Cox...")
    cph_multi, multi_sig, multi_top = run_multivariate_cox(df_mortality, "Follow Up Data", "Total mortality")
    plot_feature_importance(multi_top, title="Multivariate Cox Feature Importance")
    plot_forest(multi_sig, title="Multivariate Cox Forest Plot")

    print("Running DeepSurv...")
    ds_model, ds_surv, ds_ev = train_deepsurv(df_mortality, "Follow Up Data", "Total mortality")

    # ---------------------------
    # Cardiovascular Mortality
    # ---------------------------
    print("\n--- Cardiovascular Mortality ---")
    df_cv = prepare_cardiovascular_data(df_main)
    
    print("Running Competing Risks Cox...")
    cph_cv, cph_other_cv, comp_cv = run_competing_risks_cox(df_cv, event1_val=1, event2_val=2)
    plot_competing_risks_comparison(comp_cv, label1="CV death", label2="Other death")

    print("Running DeepHit...")
    dh_model, dh_surv, dh_ev = train_deephit_competing_risks(df_cv, "Follow Up Data", "death")
    plot_cif(dh_surv, dh_ev, event_types=[1, 2])

    # ---------------------------
    # MI (Myocardial Infarction)
    # ---------------------------
    print("\n--- Myocardial Infarction ---")
    df_mi = prepare_mi_data(df_main)
    print("Running Competing Risks Cox for MI...")
    cph_mi, cph_other_mi, comp_mi = run_competing_risks_cox(df_mi, event1_val=1, event2_val=2, duration_col="MI_date", event_col="events")
    plot_competing_risks_comparison(comp_mi, label1="MI event", label2="Other event")

    # ---------------------------
    # Stroke (Ictus)
    # ---------------------------
    print("\n--- Stroke (Ictus) ---")
    df_stroke = prepare_stroke_data(df_main)
    print("Running Competing Risks Cox for Stroke...")
    cph_stroke, cph_other_stroke, comp_stroke = run_competing_risks_cox(df_stroke, event1_val=1, event2_val=2, duration_col="Ictus_date", event_col="events")
    plot_competing_risks_comparison(comp_stroke, label1="Stroke event", label2="Other event")

    # ---------------------------
    # Other Datasets
    # ---------------------------
    print("\n--- Other Datasets Checking ---")
    # Uncomment to load and check URRAH dataset
    # df_urrah, df_urrah_legend = load_urrah_dataset()
    
if __name__ == "__main__":
    main()
