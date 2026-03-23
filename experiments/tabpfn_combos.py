"""
tabpfn_combos.py — Systematic 6×6 TabPFN × Survival-Head experiment matrix.

Usage (via uv run):
    uv run python experiments/tabpfn_combos.py --dataset GBSG --folds 5
    uv run python experiments/tabpfn_combos.py --dataset sirbu --folds 5
    uv run python experiments/tabpfn_combos.py --dataset METABRIC --folds 3
    uv run python experiments/tabpfn_combos.py --dataset SUPPORT2 --folds 5

Embedding variants (X axis):
    raw                — no TabPFN; raw features only (baseline)
    tabpfn_vanilla     — TabPFN with n_fold=0 (existing embedding_cox.py approach)
    tabpfn_kfold3      — TabPFN with n_fold=3 out-of-fold embeddings
    tabpfn_kfold5      — TabPFN with n_fold=5 out-of-fold embeddings
    tabpfn_retrieval_k10  — TabPFN with top-10 nearest-neighbour retrieval context
    tabpfn_retrieval_k50  — TabPFN with top-50 nearest-neighbour retrieval context

Survival head variants (Y axis):
    cox           — Cox Proportional Hazards (lifelines)
    deephit       — DeepHit discrete-time (pycox DeepHitSingle)
    deephit_single — DeepHit single-risk (alias, same implementation)
    pchazard      — PC-Hazard / Logistic-Hazard (pycox)
    mtlr          — MTLR (pycox, if available)
    rsf           — Random Survival Forest (sksurv, non-deep baseline)

Results are written to results/tabpfn_combos_<dataset>.csv.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from lifelines import CoxPHFitter
from pycox.evaluation import EvalSurv
from pycox.models import DeepHitSingle, LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# TabPFN import — graceful fallback if package not installed
try:
    import torch as _torch
    from survpfn.tabpfn.utils import TabPFNClassifier

    _TABPFN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TABPFN_AVAILABLE = False
    warnings.warn(
        "tabpfn package not found. TabPFN embedding variants will be skipped.",
        ImportWarning,
        stacklevel=2,
    )

# MTLR import — optional pycox model
try:
    from pycox.models import MTLR as _MTLR

    _MTLR_AVAILABLE = True
except (ImportError, AttributeError):
    _MTLR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_VARIANTS = [
    "raw",
    "tabpfn_vanilla",
    "tabpfn_kfold3",
    "tabpfn_kfold5",
    "tabpfn_retrieval_k10",
    "tabpfn_retrieval_k50",
]

SURVIVAL_HEAD_VARIANTS = [
    "cox",
    "deephit",
    "deephit_single",
    "pchazard",
    "mtlr",
    "rsf",
]


def _build_tabpfn_classifier() -> "TabPFNClassifier":
    """Instantiate a TabPFNClassifier on the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier(n_estimators=1, device=device)


def _cosine_retrieval_indices(
    X_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return indices of the top-k most cosine-similar training samples for each query.

    Parameters
    ----------
    X_train:
        Training feature matrix, shape (n_train, n_features).
    X_query:
        Query feature matrix, shape (n_query, n_features).
    k:
        Number of nearest neighbours to retrieve per query.

    Returns
    -------
    np.ndarray
        Integer index array of shape (n_query, k).
    """
    # L2-normalise rows so dot product == cosine similarity
    eps = 1e-10
    X_tr_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
    X_q_norm = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + eps)
    # (n_query, n_train)
    sims = X_q_norm @ X_tr_norm.T
    # Descending sort; take top-k
    top_k = np.argsort(-sims, axis=1)[:, :k]
    return top_k


def get_embedding(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings for train and test splits according to *variant*.

    Parameters
    ----------
    X_train:
        Training feature matrix (n_train, n_features), already scaled.
    y_train:
        Binary event labels for the training set (used as proxy classification
        target for TabPFN context).
    X_test:
        Test feature matrix (n_test, n_features), already scaled.
    variant:
        One of :data:`EMBEDDING_VARIANTS`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X_train_emb, X_test_emb)`` — embedded feature matrices.

    Raises
    ------
    ValueError
        If *variant* is not recognised.
    ImportError
        If a TabPFN variant is requested but the package is not installed.
    """
    if variant == "raw":
        return X_train.copy(), X_test.copy()

    if not _TABPFN_AVAILABLE:
        raise ImportError(
            f"Embedding variant '{variant}' requires the tabpfn package "
            "(pip install tabpfn). It is not currently installed."
        )

    if variant == "tabpfn_vanilla":
        clf = _build_tabpfn_classifier()
        clf.fit(X_train, y_train)
        train_emb = clf.get_embeddings(X_train, data_source="train")[0]
        test_emb = clf.get_embeddings(X_test, data_source="test")[0]
        return train_emb, test_emb

    if variant in ("tabpfn_kfold3", "tabpfn_kfold5"):
        n_fold = 3 if variant == "tabpfn_kfold3" else 5
        from survpfn.tabpfn.embedding import TabPFNEmbedding

        clf = _build_tabpfn_classifier()
        extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
        train_emb = extractor.get_embeddings(
            X_train, y_train, X_train, data_source="train"
        )
        test_emb = extractor.get_embeddings(
            X_train, y_train, X_test, data_source="test"
        )
        return train_emb, test_emb

    if variant in ("tabpfn_retrieval_k10", "tabpfn_retrieval_k50"):
        k = 10 if variant == "tabpfn_retrieval_k10" else 50
        # For each test point, select its k nearest training neighbours as context.
        top_k_idx = _cosine_retrieval_indices(X_train, X_test, k=k)

        clf = _build_tabpfn_classifier()
        test_embeddings: list[np.ndarray] = []

        for i, neighbour_idx in enumerate(top_k_idx):
            X_ctx = X_train[neighbour_idx]
            y_ctx = y_train[neighbour_idx]
            clf.fit(X_ctx, y_ctx)
            # Embed the single query point given its retrieval context
            emb_i = clf.get_embeddings(
                X_test[i : i + 1], data_source="test"
            )[0]
            test_embeddings.append(emb_i)

        test_emb = np.concatenate(test_embeddings, axis=0)

        # Train embeddings: for each train point use its k nearest *other* train
        # neighbours as context (leave-one-out retrieval).
        train_embeddings: list[np.ndarray] = []
        top_k_train_idx = _cosine_retrieval_indices(X_train, X_train, k=k + 1)

        for i, neighbour_idx in enumerate(top_k_train_idx):
            # Exclude the point itself (it will be the closest match at index 0)
            ctx_idx = [j for j in neighbour_idx if j != i][:k]
            X_ctx = X_train[ctx_idx]
            y_ctx = y_train[ctx_idx]
            clf.fit(X_ctx, y_ctx)
            emb_i = clf.get_embeddings(
                X_train[i : i + 1], data_source="test"
            )[0]
            train_embeddings.append(emb_i)

        train_emb = np.concatenate(train_embeddings, axis=0)
        return train_emb, test_emb

    raise ValueError(
        f"Unknown embedding variant '{variant}'. "
        f"Choose from: {EMBEDDING_VARIANTS}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Survival head training
# ─────────────────────────────────────────────────────────────────────────────


def train_survival_head(
    X_train_emb: np.ndarray,
    durations_train: np.ndarray,
    events_train: np.ndarray,
    X_test_emb: np.ndarray,
    head_variant: str,
    num_durations: int = 100,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_nodes: list[int] | None = None,
    dropout: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a survival model on embedded features and return predictions.

    Parameters
    ----------
    X_train_emb:
        Training embedding matrix (n_train, emb_dim).
    durations_train:
        Follow-up times for the training set.
    events_train:
        Binary event indicators for the training set (1=event, 0=censored).
    X_test_emb:
        Test embedding matrix (n_test, emb_dim).
    head_variant:
        One of :data:`SURVIVAL_HEAD_VARIANTS`.
    num_durations:
        Number of discrete time bins for discrete-time models.
    epochs:
        Training epochs for neural models.
    batch_size:
        Mini-batch size for neural models.
    lr:
        Learning rate for neural models.
    num_nodes:
        Hidden layer sizes for the MLP backbone. Defaults to ``[128, 64]``.
    dropout:
        Dropout probability for the MLP backbone.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(risk_scores, surv_probs, surv_times)`` where
        - *risk_scores*: shape ``(n_test,)``, higher = higher risk
        - *surv_probs*: shape ``(n_test, n_times)``, survival probabilities
        - *surv_times*: shape ``(n_times,)``, time grid

    Raises
    ------
    ValueError
        If *head_variant* is not recognised or a required optional dependency
        is missing.
    """
    if num_nodes is None:
        num_nodes = [128, 64]

    X_tr = X_train_emb.astype(np.float32)
    X_te = X_test_emb.astype(np.float32)
    t_tr = durations_train.astype(np.float32)
    e_tr = events_train.astype(np.float32)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # ── Cox PH (lifelines) ────────────────────────────────────────────────────
    if head_variant == "cox":
        df_train_cox = pd.DataFrame(X_tr, columns=[f"f{i}" for i in range(X_tr.shape[1])])
        df_train_cox["duration"] = t_tr
        df_train_cox["event"] = e_tr.astype(int)

        cph = CoxPHFitter(penalizer=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df_train_cox, duration_col="duration", event_col="event")

        df_test_cox = pd.DataFrame(X_te, columns=[f"f{i}" for i in range(X_te.shape[1])])
        risk_scores = cph.predict_partial_hazard(df_test_cox).values

        # Survival function on a shared time grid
        t_grid = np.percentile(t_tr[e_tr == 1], np.linspace(1, 99, 100))
        surv_df = cph.predict_survival_function(df_test_cox, times=t_grid)
        surv_probs = surv_df.values.T  # (n_test, n_times)
        surv_times = t_grid
        return risk_scores, surv_probs, surv_times

    # ── DeepHit / DeepHitSingle ───────────────────────────────────────────────
    if head_variant in ("deephit", "deephit_single"):
        labtrans = LabTransDiscreteTime(num_durations)
        y_tr_dh = labtrans.fit_transform(t_tr, e_tr)

        in_features = X_tr.shape[1]
        out_features = labtrans.out_features
        net = tt.practical.MLPVanilla(
            in_features, num_nodes, out_features=out_features,
            batch_norm=True, dropout=dropout, activation=nn.ReLU,
        )
        optimizer = tt.optim.AdamWR(lr=lr)
        model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
        model.fit(X_tr, y_tr_dh, batch_size=batch_size, epochs=epochs, verbose=False)

        surv_df = model.predict_surv_df(X_te)
        surv_times = surv_df.index.values
        surv_probs = surv_df.values.T  # (n_test, n_times)
        ev = EvalSurv(surv_df, durations_train, events_train, censor_surv="km")
        risk_scores = 1.0 - surv_df.iloc[-1].values  # prob of event by last time
        return risk_scores, surv_probs, surv_times

    # ── PC-Hazard / Logistic-Hazard ───────────────────────────────────────────
    if head_variant == "pchazard":
        labtrans = LogisticHazard.label_transform(num_durations)
        y_tr_lh = labtrans.fit_transform(t_tr, e_tr)

        in_features = X_tr.shape[1]
        out_features = labtrans.out_features
        net = tt.practical.MLPVanilla(
            in_features, num_nodes, out_features=out_features,
            batch_norm=True, dropout=dropout, activation=nn.ReLU,
        )
        optimizer = tt.optim.AdamWR(lr=lr)
        model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts)
        model.fit(X_tr, y_tr_lh, batch_size=batch_size, epochs=epochs, verbose=False)

        surv_df = model.predict_surv_df(X_te)
        surv_times = surv_df.index.values
        surv_probs = surv_df.values.T
        risk_scores = 1.0 - surv_df.iloc[-1].values
        return risk_scores, surv_probs, surv_times

    # ── MTLR ─────────────────────────────────────────────────────────────────
    if head_variant == "mtlr":
        if not _MTLR_AVAILABLE:
            raise ValueError(
                "MTLR is not available in the installed version of pycox. "
                "Install a version that ships pycox.models.MTLR or skip this head."
            )
        labtrans = _MTLR.label_transform(num_durations)
        y_tr_mtlr = labtrans.fit_transform(t_tr, e_tr)

        in_features = X_tr.shape[1]
        out_features = labtrans.out_features
        net = tt.practical.MLPVanilla(
            in_features, num_nodes, out_features=out_features,
            batch_norm=True, dropout=dropout, activation=nn.ReLU,
        )
        optimizer = tt.optim.AdamWR(lr=lr)
        model = _MTLR(net, optimizer, duration_index=labtrans.cuts)
        model.fit(X_tr, y_tr_mtlr, batch_size=batch_size, epochs=epochs, verbose=False)

        surv_df = model.predict_surv_df(X_te)
        surv_times = surv_df.index.values
        surv_probs = surv_df.values.T
        risk_scores = 1.0 - surv_df.iloc[-1].values
        return risk_scores, surv_probs, surv_times

    # ── Random Survival Forest (sksurv) ───────────────────────────────────────
    if head_variant == "rsf":
        y_train_sksurv = np.array(
            [(bool(e), float(t)) for e, t in zip(e_tr, t_tr)],
            dtype=[("event", bool), ("time", float)],
        )
        rsf = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=random_state,
        )
        rsf.fit(X_tr, y_train_sksurv)

        risk_scores = rsf.predict(X_te)
        surv_funcs = rsf.predict_survival_function(X_te)
        surv_times = rsf.unique_times_
        surv_probs = np.stack([fn(surv_times) for fn in surv_funcs])  # (n_test, n_times)
        return risk_scores, surv_probs, surv_times

    raise ValueError(
        f"Unknown survival head variant '{head_variant}'. "
        f"Choose from: {SURVIVAL_HEAD_VARIANTS}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(dataset: str) -> tuple[pd.DataFrame, str, str]:
    """Load a named dataset and return ``(df, duration_col, event_col)``.

    Supported dataset names (case-insensitive):
        sirbu    — private Sirbu cardiovascular cohort
        urrah    — private URRAH cardiovascular cohort
        GBSG     — German Breast Cancer Study Group (pycox built-in)
        METABRIC — Molecular Taxonomy of Breast Cancer (pycox built-in)
        SUPPORT2 — Study to Understand Prognoses and Preferences (pycox built-in)

    Parameters
    ----------
    dataset:
        Dataset identifier string.

    Returns
    -------
    tuple[pd.DataFrame, str, str]
        DataFrame with features + outcome columns, duration column name,
        event column name.

    Raises
    ------
    ValueError
        If *dataset* is not recognised.
    """
    name = dataset.lower()

    if name == "gbsg":
        from pycox.datasets import gbsg

        df = gbsg.read_df()
        return df, "duration", "event"

    if name == "metabric":
        from pycox.datasets import metabric

        df = metabric.read_df()
        return df, "duration", "event"

    if name == "support2":
        from pycox.datasets import support

        df = support.read_df()
        return df, "duration", "event"

    if name == "sirbu":
        # Private dataset — requires local file
        from survpfn.preprocessing import clean_and_impute, prepare_cox_data

        data_path = os.environ.get(
            "SIRBU_DATA_PATH",
            "Dataset Sirbu/Sirbu Dataset_2024.xlsx",
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Sirbu dataset not found at '{data_path}'. "
                "Set the SIRBU_DATA_PATH environment variable to the correct path."
            )
        df_raw = pd.read_excel(data_path)
        df_clean = clean_and_impute(df_raw)
        df = prepare_cox_data(df_clean)
        return df, "Follow Up Data", "Total mortality"

    if name == "urrah":
        from survpfn.other_datasets import load_urrah_dataset

        urrah_path = os.environ.get(
            "URRAH_DATA_PATH",
            "Dataset Sirbu/URRAH_TG_conLegenda.xlsx",
        )
        df_val, _ = load_urrah_dataset(urrah_path)
        df_val = df_val.dropna(subset=["AA_SURV_MESI", "DEAD"])
        return df_val, "AA_SURV_MESI", "DEAD"

    raise ValueError(
        f"Unknown dataset '{dataset}'. "
        "Supported: sirbu, urrah, GBSG, METABRIC, SUPPORT2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation runner
# ─────────────────────────────────────────────────────────────────────────────


def run_cv(
    dataset: str,
    folds: int = 5,
    embedding_variants: list[str] | None = None,
    head_variants: list[str] | None = None,
    results_dir: str = "results",
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full TabPFN × Survival-Head combo grid under K-fold CV.

    For each (embedding_variant, head_variant) pair the function:
      1. Splits data into *folds* stratified folds.
      2. Scales features (StandardScaler fit on train fold only).
      3. Extracts embeddings via :func:`get_embedding`.
      4. Trains the survival head via :func:`train_survival_head`.
      5. Evaluates with Harrell C-index.
      6. Aggregates results across folds.

    Parameters
    ----------
    dataset:
        Dataset name passed to :func:`load_dataset`.
    folds:
        Number of cross-validation folds.
    embedding_variants:
        Subset of :data:`EMBEDDING_VARIANTS` to evaluate. Defaults to all.
    head_variants:
        Subset of :data:`SURVIVAL_HEAD_VARIANTS` to evaluate. Defaults to all.
    results_dir:
        Directory where ``tabpfn_combos_<dataset>.csv`` will be saved.
    random_state:
        Random seed for CV splitting and model initialisation.
    verbose:
        Print fold-level progress to stdout.

    Returns
    -------
    pd.DataFrame
        One row per (embedding_variant, head_variant, fold) with columns:
        ``dataset``, ``embedding``, ``head``, ``fold``, ``c_index``, ``status``.
    """
    if embedding_variants is None:
        embedding_variants = EMBEDDING_VARIANTS
    if head_variants is None:
        head_variants = SURVIVAL_HEAD_VARIANTS

    df, duration_col, event_col = load_dataset(dataset)

    # Drop rows with missing outcome
    df = df.dropna(subset=[duration_col, event_col])
    df = df.reset_index(drop=True)

    X_all = df.drop(columns=[duration_col, event_col]).values.astype(np.float32)
    t_all = df[duration_col].values.astype(np.float64)
    e_all = df[event_col].values.astype(np.float32)

    # Stratification label: event × time-quartile
    time_quartile = pd.qcut(t_all, q=4, labels=False, duplicates="drop")
    strata = e_all.astype(int) * 4 + time_quartile

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, strata)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx + 1}/{folds}  |  Dataset: {dataset}")
            print(f"{'='*60}")

        X_tr_raw = X_all[train_idx]
        X_te_raw = X_all[test_idx]
        t_tr = t_all[train_idx].astype(np.float32)
        e_tr = e_all[train_idx]
        t_te = t_all[test_idx]
        e_te = e_all[test_idx]

        # Feature scaling: fit on train fold only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_te_scaled = scaler.transform(X_te_raw)

        # Binary event label for TabPFN proxy-classification task
        y_tr_binary = e_tr.astype(int)

        for emb_variant in embedding_variants:
            for head_variant in head_variants:
                combo_key = f"{emb_variant} × {head_variant}"
                if verbose:
                    print(f"  Running: {combo_key} ...", end=" ", flush=True)

                status = "ok"
                c_index = float("nan")

                try:
                    # Step 1: extract embeddings
                    X_tr_emb, X_te_emb = get_embedding(
                        X_tr_scaled,
                        y_tr_binary,
                        X_te_scaled,
                        variant=emb_variant,
                    )

                    # Step 2: train survival head
                    risk_scores, _surv_probs, _surv_times = train_survival_head(
                        X_tr_emb,
                        t_tr,
                        e_tr,
                        X_te_emb,
                        head_variant=head_variant,
                    )

                    # Step 3: evaluate
                    c_index_result = concordance_index_censored(
                        e_te.astype(bool),
                        t_te,
                        risk_scores,
                    )
                    c_index = float(c_index_result[0])

                    if verbose:
                        print(f"C-index = {c_index:.4f}")

                except (ImportError, ValueError, RuntimeError, Exception) as exc:
                    status = f"error: {type(exc).__name__}: {exc}"
                    if verbose:
                        print(f"FAILED — {status}")

                records.append(
                    {
                        "dataset": dataset,
                        "embedding": emb_variant,
                        "head": head_variant,
                        "fold": fold_idx + 1,
                        "c_index": c_index,
                        "status": status,
                    }
                )

    results_df = pd.DataFrame(records)

    # Summary: mean ± std per combo
    summary = (
        results_df[results_df["status"] == "ok"]
        .groupby(["embedding", "head"])["c_index"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "c_index_mean", "std": "c_index_std", "count": "n_folds"})
        .reset_index()
    )
    if verbose:
        print("\n\nSummary (mean C-index over folds):")
        print(summary.to_string(index=False))

    # Persist raw fold-level results
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"tabpfn_combos_{dataset}.csv")
    results_df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nResults saved to: {out_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TabPFN × Survival-Head 6×6 experiment matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="GBSG",
        help=(
            "Dataset to evaluate. One of: sirbu, urrah, GBSG, METABRIC, SUPPORT2. "
            "Private datasets (sirbu, urrah) require local data files."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        choices=EMBEDDING_VARIANTS,
        metavar="VARIANT",
        help=(
            "Subset of embedding variants to run. "
            f"Defaults to all: {EMBEDDING_VARIANTS}"
        ),
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=None,
        choices=SURVIVAL_HEAD_VARIANTS,
        metavar="HEAD",
        help=(
            "Subset of survival head variants to run. "
            f"Defaults to all: {SURVIVAL_HEAD_VARIANTS}"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where result CSV files are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-fold progress output.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point — parse arguments and launch :func:`run_cv`."""
    args = _parse_args()
    run_cv(
        dataset=args.dataset,
        folds=args.folds,
        embedding_variants=args.embeddings,
        head_variants=args.heads,
        results_dir=args.results_dir,
        random_state=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
