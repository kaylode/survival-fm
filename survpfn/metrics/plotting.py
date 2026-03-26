"""survpfn.eval.plotting — Survival analysis visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_correlation_matrix(df, threshold=0.90):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(24,20))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "feature_correlation_matrix.pdf"), bbox_inches="tight")
    plt.close()

    corr_abs = corr_matrix.abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper.columns if any(upper[col] > threshold)]

    print("Highly correlated features:", high_corr_features)


def plot_feature_importance(importance_df, top_n=20, title="Feature Importance (|log(HR)|)"):
    plt.figure(figsize=(8, 10))
    plt.barh(importance_df["feature"], importance_df["abs_logHR"])
    plt.gca().invert_yaxis()
    plt.xlabel("|log(HR)|")
    plt.title(title)

    filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("|", "").lower() + ".pdf"
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches="tight")
    plt.close()


def plot_forest(importance_sig, title="Cox Forest Plot"):
    plt.figure(figsize=(8, len(importance_sig) * 0.5))
    plt.errorbar(
        importance_sig["HR"],
        range(len(importance_sig)),
        xerr=[
            importance_sig["HR"] - importance_sig["HR_lower"],
            importance_sig["HR_upper"] - importance_sig["HR"]
        ],
        fmt='o',
        color='black',
        ecolor='gray',
        capsize=3
    )
    plt.yticks(range(len(importance_sig)), importance_sig["feature"])
    plt.axvline(x=1, color='red', linestyle='--')
    plt.xlabel("Hazard Ratio (HR)")
    plt.title(title)
    plt.gca().invert_yaxis()

    filename = title.replace(" ", "_").lower() + ".pdf"
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches="tight")
    plt.close()


def plot_competing_risks_comparison(comparison_df, label1="Event 1", label2="Event 2"):
    print(comparison_df)

    plot_df = comparison_df[["Event 1 HR", "Event 2 HR"]].rename(
        columns={"Event 1 HR": f"{label1} HR", "Event 2 HR": f"{label2} HR"}
    )
    plot_df.plot.bar(figsize=(10,6))

    plt.title(f"{label1} vs {label2} HR Comparison")
    plt.ylabel("Hazard Ratio")

    filename = f"{label1}_vs_{label2}_hr.pdf".replace(" ", "_").lower()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches="tight")
    plt.close()


def plot_cif(surv, eval_surv, event_types=[1, 2]):
    times = surv.index.values
    for i, event_type in enumerate(event_types):
        mask = eval_surv.events == event_type
        cif = 1 - surv.loc[:, mask]
        plt.step(times, cif.mean(axis=1), label=f"CIF Event {event_type}")

    plt.xlabel("Time")
    plt.ylabel("Cumulative Incidence")
    plt.title("Cumulative Incidence Function (CIF)")
    plt.legend()

    plt.savefig(os.path.join(RESULTS_DIR, "cumulative_incidence_function.pdf"), bbox_inches="tight")
    plt.close()
