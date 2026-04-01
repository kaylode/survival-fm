import faiss
import numpy as np
import torch


class FAISS:
    """
    This class initializes a FAISS index with the provided data and allows for efficient nearest neighbor search.
    """

    def __init__(
        self, X: np.ndarray, use_hnsw: bool = False, hnsw_m: int = 32, metric: str = "L2"
    ) -> None:
        """
        Initializes the FAISS index with the provided data.
        Args:
            X (np.ndarray): The data to index, shape should be (n_samples, n_features).
            use_hnsw (bool): Whether to use HNSW index or not.
            hnsw_m (int): The number of bi-directional links created for each element in the HNSW index.
            metric (str): The distance metric to use, either "L2" for Euclidean distance or "IP" for inner product.
        """
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(X.shape[1], hnsw_m)
            if metric == "L2":
                self.index.metric_type = faiss.METRIC_L2
            elif metric == "IP":
                self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise NotImplementedError
        else:
            if metric == "L2":
                self.index = faiss.IndexFlatL2(X.shape[1])
            elif metric == "IP":
                self.index = faiss.IndexFlatIP(X.shape[1])
            else:
                raise NotImplementedError
        self.index.add(X)

    def get_knn_indices(self, queries: np.ndarray | torch.Tensor, k: int) -> np.ndarray:
        """retreive the k-nearest neighbors indices for the given queries.

        Args:
            queries (np.ndarray|torch.Tensor): query points for which to find nearest neighbors.
            k (int): number of nearest neighbors to retrieve.

        Returns:
            np.ndarray: k nearest neighbors indices for each query point.
        """
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs
