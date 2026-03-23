"""
Retrieval-Augmented TabPFN for Survival Analysis.

Strategy: for each test sample, retrieve the K most similar training samples
(by cosine/Euclidean distance in feature space) and use them as TabPFN context.
This solves TabPFN's N>1000 scalability limit for large cohorts like URRAH.

TODO: implement full retrieval-augmented embedding extraction.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_retrieval_context(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 50,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each test sample, retrieve the K nearest training samples.
    Returns per-test-sample context arrays of shape (n_test, K, n_features).

    Args:
        X_train: (n_train, n_features)
        y_train: (n_train,) binary event labels
        X_test:  (n_test, n_features)
        k:       context size
        metric:  "cosine" or "euclidean"

    Returns:
        ctx_X: (n_test, K, n_features) — retrieved feature contexts
        ctx_y: (n_test, K)             — retrieved label contexts
        indices: (n_test, K)           — indices into X_train
    """
    raise NotImplementedError("Retrieval-augmented TabPFN — see experiments/tabpfn_combos.py for current implementation")
