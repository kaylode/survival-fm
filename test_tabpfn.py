import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


from src.data_loader import load_and_merge_data
from src.preprocessing import clean_and_impute, prepare_cox_data
from src.tabpfn import (
	get_tabpfn_embeddings,
	setup_figure, create_savefig_partial
)

def main():
    # 1. Load and clean the data
    data = load_and_merge_data("Dataset Sirbu")
    df_main = clean_and_impute(data)
    
    # 2. Extract specific features and targets (e.g. Mortality data)
    df_mortality = prepare_cox_data(df_main)
    
    X = df_mortality.drop(columns=["Follow Up Data", "Total mortality"])
    y = df_mortality["Total mortality"]
    
    # 3. Split into Train & Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )
    
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # 4. Generate the embeddings!
    print("Generating TabPFN Embeddings...")
    embeddings = get_tabpfn_embeddings(X_train, y_train, X_test, y_test)
    print(f"Successfully generated embeddings with shape: {embeddings.shape}")

    # 5. Visualize embeddings via t-SNE
    if embeddings.ndim == 3:
        embeddings = embeddings[0]

    print(f"Running t-SNE on {embeddings.shape} points...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(embeddings)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = setup_figure(figsize=(8, 8), style='seaborn-v0_8-paper')
    
    palette = ['#1f77b4', '#d62728'] # Blue for negatives, Red for positives
    labels = {0: 'Negative (Alive)', 1: 'Positive (Deceased)'}
    
    unique_classes = np.unique(y_test)
    for i, cls in enumerate(unique_classes):
        mask = (y.values == cls)
        ax.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1], 
            label=labels.get(cls, str(cls)),
            s=120, 
            color=palette[i % len(palette)], 
            alpha=0.6, 
            edgecolors='black', 
            linewidths=0.1
        )
        
    ax.set_xlabel("t-SNE Component 1", fontsize=24, fontweight='bold')
    ax.set_ylabel("t-SNE Component 2", fontsize=24, fontweight='bold')
    
    ax.legend(
        bbox_to_anchor=(1.0, 1.0), 
        loc='upper right', 
        title="Mortality Status", 
        frameon=True, 
        shadow=True,
        fontsize=10
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    
    out_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    save_name_base = "tabpfn_mortality_tsne"
    savefig = create_savefig_partial(
        fig_dir=out_dir,
        fig_fmt='pdf',
        fig_size=(12, 12),
        save=True,
        dpi=300
    )
    savefig(fig, save_name_base)
    plt.close(fig)
    print(f"Visualization saved to {os.path.join(out_dir, save_name_base)}.pdf")

if __name__ == "__main__":
    main()