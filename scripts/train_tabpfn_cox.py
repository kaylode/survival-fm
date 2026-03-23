"""
scripts/train_tabpfn_cox.py — Train the TabPFN-aware Cox PH model on the Sirbu dataset.

Usage:
    uv run python scripts/train_tabpfn_cox.py --epochs 100 --lr 1e-3 --device cuda:0
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from survpfn.data.loader import load_and_merge_data
from survpfn.data.preprocessing import clean_and_impute, prepare_cox_data
from survpfn.models.tabpfn.cox import TabPFNCoxPH
from survpfn.eval.metrics import evaluate_survival_model


def train_aware_cox_model(args):
    # Determine torch dtype from string
    dtype = torch.float32
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16

    # 1. Load and clean the data
    print("Loading and cleaning data...")
    raw_data = load_and_merge_data("Dataset Sirbu")
    df_main = clean_and_impute(raw_data)

    # 2. Extract specific task data (e.g. Total Mortality)
    print("Preparing task-specific data...")
    df_task = prepare_cox_data(df_main)

    # Define columns
    duration_col = "Follow Up Data"
    event_col = "Total mortality"

    X = df_task.drop(columns=[duration_col, event_col])
    # Total mortality (0=alive, 1=dead) will be our PFN labels
    y_pfn = df_task[event_col].values.astype(np.int64)
    durations = df_task[duration_col].values.astype(np.float32)
    events = df_task[event_col].values.astype(np.float32)

    # 3. Split into Train & Test sets
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=events
    )

    X_train_raw, X_test_raw = X.iloc[idx_train], X.iloc[idx_test]
    y_pfn_train, y_pfn_test = y_pfn[idx_train], y_pfn[idx_test]
    dur_train, dur_test = durations[idx_train], durations[idx_test]
    ev_train, ev_test = events[idx_train], events[idx_test]

    # 4. Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Note: TabPFN usually expects a fixed number of features (e.g. 100).
    # If your data has more features, we take the first N to match TabPFN's encoder expectation.
    # Alternatively, you can use PCA or feature selection.

    # 5. Initialize TabPFNCoxPH
    # We use n_out=2 for binary mortality classification in the PFN path
    print("Initializing TabPFNCoxPH model...")
    model = TabPFNCoxPH(
        n_out=2,
        head_num_nodes=[128, 64],
        learning_rate=args.lr,
        alpha=args.alpha,
        dropout=0.2,
        freeze_tabpfn=True,  # always freeze tabpfn
        dtype=dtype,
        device=args.device
    )

    # Check if we need to truncate features for TabPFN
    pfn_num_features = model.net.tabpfn.encoder.in_features if hasattr(model.net.tabpfn.encoder, 'in_features') else 100
    if X_train.shape[1] > pfn_num_features:
        print(f"Truncating features from {X_train.shape[1]} to {pfn_num_features} to match TabPFN encoder...")
        X_train = X_train[:, :pfn_num_features]
        X_test = X_test[:, :pfn_num_features]

    # 6. Fit the model
    print("\nStarting training (Joint Cox + PFN Loss)...")
    model.fit(
        x=X_train,
        durations=dur_train,
        events=ev_train,
        y_pfn=y_pfn_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True
    )

    # 7. Evaluate
    print("\n--- Evaluation ---")
    model.net.eval()

    # Compute baseline hazards for survival probabilities
    print("Computing baseline hazards...")
    # pycox objects need the training data to compute baseline hazards
    X_train_pt = torch.from_numpy(X_train).to(args.device, dtype)
    dur_train_pt = torch.from_numpy(dur_train).to(args.device, dtype)
    ev_train_pt = torch.from_numpy(ev_train).to(args.device, dtype)

    with torch.no_grad():
        # Ensure we compute baseline hazards on training set
        model.model.compute_baseline_hazards(input=X_train_pt, target=(dur_train_pt, ev_train_pt))

    # Predict risk scores and survival probabilities for test set
    print("Predicting risk scores and survival probabilities...")
    X_test_pt = torch.from_numpy(X_test).to(args.device, dtype)
    with torch.no_grad():
        risk_scores = model.predict_survival(X_test)
        # surv_df shape: (n_times, n_samples)
        surv_df = model.model.predict_surv_df(X_test_pt)
        surv_probs = surv_df.values.T  # (n_samples, n_times)
        surv_times = surv_df.index.values

    # Prepare dataframes for evaluate_survival_model
    y_train_df = pd.DataFrame({duration_col: dur_train, event_col: ev_train})
    y_test_df = pd.DataFrame({duration_col: dur_test, event_col: ev_test})

    metrics = evaluate_survival_model(
        y_train=y_train_df,
        y_test=y_test_df,
        duration_col=duration_col,
        event_col=event_col,
        risk_scores=risk_scores,
        surv_probs=surv_probs,
        surv_times=surv_times
    )

    print("\nFinal Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
             print(f"  {k}: {v:.4f}")
        else:
             print(f"  {k}: {v}")

    # Optional: Save results
    os.makedirs("results", exist_ok=True)
    pd.Series(metrics).to_csv("results/tabpfn_cox_metrics.csv")
    print(f"\nEvaluation complete. Results saved to results/tabpfn_cox_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabPFN Survival Model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for PFN loss")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Torch dtype")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computing device")

    args = parser.parse_args()

    train_aware_cox_model(args)
