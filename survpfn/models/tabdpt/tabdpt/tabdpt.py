import math
from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch.nn.attention import SDPBackend, sdpa_kernel

from .model import TabDPTModel
from .models_utils import Task, convert_to_torch_tensor, pad_x
from .retrieval import FAISS

_DEFAULT_DEVICE = "cuda:0"
_INF_BATCH_SIZE = 512
_DEFAULT_CONTEXT_SIZE = 128
_DEFAULT_RETRIEVAL = True
_DEFAULT_TEMPERATURE = 0.8


class TabDPTEstimator(BaseEstimator):
    """A PyTorch-based implementation of TabDPT for tabular data tasks."""

    def __init__(
        self,
        model: Optional[TabDPTModel] = None,
        path: str = "",
        mode: Task = Task.CLS,
        inf_batch_size: int = _INF_BATCH_SIZE,
        device: str = _DEFAULT_DEVICE,
        tensor_eval: bool = False,
    ):
        """Initialize the TabDPTEstimator.

        Args:
            model (Optional[TabDPTModel], optional): The TabDPT model to use. Defaults to None.
            path (str, optional): Path to the model checkpoint. Defaults to "".
            mode (Task, optional): The task mode (classification or regression). Defaults to Task.CLS.
            inf_batch_size (int, optional): Inference batch size. Defaults to 512.
            device (str, optional): Device to run the model on. Defaults to _DEFAULT_DEVICE.
            tensor_eval (bool, optional): Whether to use tensor evaluation. Defaults to False.

        Raises:
            ValueError: If both model and path are None.
        """
        # seed_everything(42)
        self.mode = mode
        self.inf_batch_size = inf_batch_size
        self.device = device
        self.tensor_eval = tensor_eval
        if model is None:
            if path:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.model = TabDPTModel.load(
                    model_state=checkpoint["model"], config=checkpoint["cfg"]
                )
                self.model.to(self.device)
                self.model.eval()
            else:
                raise ValueError("Either model or path must be provided")
        else:
            self.model = model
        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out

    def fit(self, X: np.ndarray, y: np.ndarray, faiss_index=None) -> None:
        """Fit the model to the training data.

        Args:
            X (np.ndarray): Training features, a 2D numpy array of shape [n_samples, n_features].
            y (np.ndarray): Training labels, a 1D numpy array of shape [n_samples].
            faiss_index (Optional[FAISS], optional): Precomputed FAISS index for retrieval. Defaults to None.
        """
        if not self.tensor_eval:
            assert isinstance(X, np.ndarray), "X must be a numpy array"
            assert isinstance(y, np.ndarray), "y must be a numpy array"
            assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
            assert X.ndim == 2, "X must be a 2D array"
            assert y.ndim == 1, "y must be a 1D array"

        # initialize the Faiss index if not provided
        self.faiss_knn = faiss_index if faiss_index is not None else FAISS(X)

        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y

        self.autocast = torch.autocast(
            device_type="cuda" if "cuda" in str(self.device) else "cpu",
            dtype=torch.bfloat16 if "cuda" in str(self.device) else torch.float16
        )

    def _prepare_prediction(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """preprocess the input data for prediction.
        This method handles the transformation of the input data, including imputation,
        scaling, and dimensionality reduction if necessary.

        Args:
            X (np.ndarray): input features for prediction.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: training features, training labels, and test features.
        """

        # preprocess the input data
        self.X_test = X

        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA optionally to reduce the number of features
        if self.n_features > self.max_features:
            _, _, self.V = torch.pca_lowrank(train_x, q=self.max_features)
            train_x = train_x @ self.V
        else:
            self.V = None

        # apply PCA to the test set if V is not None (i.e., PCA was applied to the training set)
        test_x = test_x @ self.V if self.V is not None else test_x

        return train_x, train_y, test_x

    @torch.no_grad()
    def no_retrieval_data(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for the no retrieval scenario.

        Args:
            train_x (torch.Tensor): Training features with shape [n_train, d].
            train_y (torch.Tensor): Training labels with shape [n_train].
            test_x (torch.Tensor): Test features with shape [n_test, d].
            context_size (int, optional): Number of context samples to use.
                Defaults to _DEFAULT_CONTEXT_SIZE.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Context features, context labels, and evaluation features.
        """
        n_context = min(context_size, train_x.shape[0])
        idx_context = np.random.choice(train_x.shape[0], n_context, replace=False)

        # shape: [n_context, d]
        context_features = train_x[idx_context]
        context_labels = train_y[idx_context]

        # shape: [n_context, 1, d]
        context_features = pad_x(context_features.unsqueeze(1), self.max_features).to(self.device)
        context_labels = context_labels.unsqueeze(1).float()  # shape: [n_context, 1]

        # shape: [n_test, 1, d]
        eval_features = pad_x(test_x.unsqueeze(1), self.max_features).to(self.device)

        return context_features, context_labels, eval_features

    @torch.no_grad()
    def batch_retrieval_data(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        batch_index: int,
        context_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for the batch retrieval scenario.

        Args:
            batch_index (int): Batch index.
            context_size (int): Number of context samples to use.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Context features, context labels, and evaluation features.
        """
        n_test = test_x.shape[0]

        start = batch_index * self.inf_batch_size
        end = min(n_test, (batch_index + 1) * self.inf_batch_size)

        indices_knn = self.faiss_knn.get_knn_indices(self.X_test[start:end], k=context_size)
        knn_features = train_x[torch.tensor(indices_knn)]
        knn_labels = train_y[torch.tensor(indices_knn)]

        # swap => [context_size, batch_size, d]
        knn_features = np.swapaxes(knn_features.cpu().numpy(), 0, 1)
        knn_labels = np.swapaxes(knn_labels.cpu().numpy(), 0, 1)

        knn_features = pad_x(torch.Tensor(knn_features), self.max_features).to(self.device)
        knn_labels = torch.Tensor(knn_labels).to(self.device)

        eval_features = pad_x(test_x[start:end].unsqueeze(0), self.max_features).to(self.device)
        return knn_features, knn_labels, eval_features


class TabDPTClassifier(TabDPTEstimator, ClassifierMixin):
    """
    A PyTorch-based implementation of TabDPT for classification tasks.
    """

    def __init__(
        self,
        model: Optional[TabDPTModel] = None,
        path: str = "",
        mode: Task = Task.CLS,
        inf_batch_size: int = _INF_BATCH_SIZE,
        device: str = _DEFAULT_DEVICE,
        tensor_eval: bool = False,
    ):
        super().__init__(
            model=model,
            path=path,
            mode=mode,
            inf_batch_size=inf_batch_size,
            device=device,
            tensor_eval=tensor_eval,
        )

    def fit(self, X, y, faiss_index=None):
        super().fit(X, y, faiss_index)
        # Number of classes
        if self.tensor_eval:
            self.num_classes = len(torch.unique(self.y_train))
        else:
            self.num_classes = len(np.unique(self.y_train))
        assert self.num_classes > 1, "Number of classes must be greater than 1"

    def _predict_large_cls(self, X_context, X_eval, y_context) -> torch.Tensor:
        """Digit-level prediction for the case where num_classes > self.max_num_classes.
        Here, X_context + X_eval is [L, 1, d], and y_context is [L_context, 1].

        Args:
            X_context (torch.Tensor): Context features with shape [L_context, 1, d].
            X_eval (torch.Tensor): Evaluation features with shape [L_eval, 1, d].
            y_context (torch.Tensor): Context labels with shape [L_context, 1].

        Returns:
            torch.Tensor: Predicted probabilities with shape [L_eval, 1, num_classes].
        """
        # number of digits needed to represent num_classes
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_context_digit = (y_context // (self.max_num_classes**i)) % self.max_num_classes

            with self.autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                pred = self.model(
                    x=torch.cat([X_context, X_eval], dim=0),
                    y=y_context_digit,
                )
            # shape: [L_context + L_eval, 1, max_num_classes]
            digit_preds.append(pred[..., : self.max_num_classes].float())

        # Combine digit predictions
        B = X_context.shape[1]
        L_eval = X_eval.shape[0]
        full_pred = torch.zeros((L_eval, B, self.num_classes), device=X_context.device)
        # For each of the L_eval positions, compute the class probabilities
        for class_idx in range(self.num_classes):
            # sum across digits
            class_pred = torch.zeros((L_eval, B), device=X_eval.device)
            for digit_idx, digit_pred in enumerate(digit_preds):
                digit_value = (
                    class_idx // (self.max_num_classes**digit_idx)
                ) % self.max_num_classes
                # digit_pred shape: [L_context + L_eval, 1, max_num_classes]
                # The last L_eval rows correspond to the actual predictions
                class_pred += digit_pred[-L_eval:, :, digit_value]  # shape: [L_eval]

            full_pred[:, :, class_idx] = class_pred
        return full_pred  # shape: [L_eval, 1, self.num_classes]

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = _DEFAULT_TEMPERATURE,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        use_retrieval: bool = _DEFAULT_RETRIEVAL,
    ) -> np.ndarray:
        """predict class probabilities for the input data.


        Args:
            X (np.ndarray): input features for prediction.
            temperature (float, optional): output probability temperature. Defaults to 0.8.
            context_size (int, optional): context size. Defaults to 128.
            use_retrieval (bool, optional):
                - If `use_retrieval=True`, do batch-by-batch retrieval.
                - If `use_retrieval=False`, pick one random context and feed all test points in one shot:
                [N_context + N_test, 1, d]. Defaults to True.

        Returns:
            np.ndarray: probbilities for each class for each test instance.
        """
        train_x, train_y, test_x = self._prepare_prediction(X)
        n_test = test_x.shape[0]

        # 1) If context_size >= entire training set => use them all
        if context_size >= self.n_instances:
            # shape: [n_train, 1, d]
            X_context = pad_x(train_x[:, None, :], self.max_features).to(self.device)
            y_context = train_y[:, None].float()
            # shape: [n_test, 1, d]
            X_eval = pad_x(test_x[:, None, :], self.max_features).to(self.device)

            if self.num_classes <= self.max_num_classes:
                with self.autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    pred = self.model(
                        x=torch.cat([X_context, X_eval], dim=0),
                        y=y_context,
                    )
                # shape: [n_train + n_test, 1, max_num_classes]
                pred = pred[..., : self.num_classes].float()
                # extract the last n_test rows => the test predictions
                test_preds = pred[-n_test:, 0, :]  # shape: [n_test, num_classes]

            else:
                # Large-class approach
                test_preds = self._predict_large_cls(X_context, X_eval, y_context)
                test_preds = test_preds.squeeze(1)  # shape: [n_test, num_classes]

            test_preds /= temperature
            test_preds = torch.nn.functional.softmax(test_preds, dim=-1)
            return test_preds.detach().cpu().numpy()

        # TODO: combine retrieval step with that of training
        # 2) If we want retrieval
        if use_retrieval:
            preds_list = []
            # batch the test set
            for b in range(math.ceil(n_test / self.inf_batch_size)):
                X_nni, y_nni, X_eval = self.batch_retrieval_data(
                    train_x, train_y, test_x, b, context_size
                )
                # forward pass
                if self.num_classes <= self.max_num_classes:  # small-class case
                    with self.autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        pred = self.model(
                            x=torch.cat([X_nni, X_eval], dim=0),
                            y=y_nni,
                        )
                    # shape: [context_size + 1, batch_size, max_num_classes]
                    pred = pred[..., : self.num_classes].float()
                    # last row => predictions for test batch
                    batch_preds = pred[-1, :, :]  # shape: [batch_size, num_classes]
                else:
                    # large-class case
                    batch_preds_full = self._predict_large_cls(X_nni, X_eval, y_nni)
                    batch_preds = batch_preds_full[-1, :, :]

                batch_preds /= temperature
                batch_preds = torch.nn.functional.softmax(batch_preds, dim=-1)
                preds_list.append(batch_preds)

            preds_all = torch.cat(preds_list, dim=0)  # [n_test, num_classes]
            return preds_all.cpu().numpy()

        # 3) If we want NO retrieval => single pass
        #    [N_context, 1, d] + [N_test, 1, d] => [N_context + N_test, 1, d]
        else:
            X_ctx, y_ctx, X_eval = self.no_retrieval_data(train_x, train_y, test_x)

            if self.num_classes <= self.max_num_classes:
                with self.autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    pred = self.model(
                        x=torch.cat([X_ctx, X_eval], dim=0),
                        y=y_ctx,
                    )
                # shape: [n_context + n_test, 1, max_num_classes]
                pred = pred[..., : self.num_classes].float()
                # The last n_test rows => the actual test preds
                test_preds = pred[-n_test:, 0, :]  # [n_test, num_classes]
            else:
                test_preds_full = self._predict_large_cls(X_ctx, X_eval, y_ctx)
                # shape: [n_context + n_test - ???, 1, num_classes]
                # We only want the last n_test rows
                test_preds = test_preds_full[-n_test:, 0, :]

            test_preds /= temperature
            test_preds = torch.nn.functional.softmax(test_preds, dim=-1)
            return test_preds.detach().cpu().numpy()

    def predict(
        self,
        X: np.ndarray,
        temperature: float = _DEFAULT_TEMPERATURE,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        use_retrieval: bool = _DEFAULT_RETRIEVAL,
    ) -> np.ndarray:
        """predict class labels for the input data.
        This method uses the `predict_proba` method to get class probabilities and then
        returns the class with the highest probability for each instance.

        Args:
            X (np.ndarray): input features for prediction.
            temperature (float, optional): temperature for prediction. Defaults to 0.8.
            context_size (int, optional): size of context. Defaults to 128.
            use_retrieval (bool, optional): whether to use retrieval. Defaults to True.

        Returns:
            np.ndarray: class labels for each test instance.
        """
        return self.predict_proba(
            X, temperature=temperature, context_size=context_size, use_retrieval=use_retrieval
        ).argmax(axis=-1)

    @torch.no_grad()
    def predict_embeddings(
        self,
        X: np.ndarray,
        context_size: int = _DEFAULT_CONTEXT_SIZE,
        use_retrieval: bool = _DEFAULT_RETRIEVAL,
        return_full: bool = True,
    ) -> np.ndarray:
        """Extract penultimate-layer embeddings.
        
        Args:
            X (np.ndarray): input features (query).
            context_size (int): number of context samples.
            use_retrieval (bool): whether to use retrieval.
            return_full (bool): If True, returns (n_ctx + n_query, 1, ninp).
                If False, returns (n_query, ninp).
        
        Returns:
            np.ndarray: embeddings.
        """
        train_x, train_y, test_x = self._prepare_prediction(X)
        n_test = test_x.shape[0]

        # 1) Non-retrieval case
        if not use_retrieval:
            X_context, y_context, X_eval = self.no_retrieval_data(
                train_x, train_y, test_x, context_size=context_size
            )
            with self.autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                _, src = self.model(
                    x=torch.cat([X_context, X_eval], dim=0),
                    y=y_context,
                    return_embeddings=True,
                )
                # src is [ctx_size + n_test, 1, ninp]
                res_t = src.detach().to(torch.float32)
                # If the classifier has its own norm (e.g. set by extractor), use it
                if hasattr(self, 'norm') and self.norm is not None:
                    res_t = self.norm(res_t)
                
                res = res_t.cpu().numpy()
                if not return_full:
                    res = res[-n_test:, 0, :]
                return res

        # 2) Retrieval case
        else:
            embeddings = []
            for b in range(math.ceil(n_test / self.inf_batch_size)):
                X_nni, y_nni, X_eval = self.batch_retrieval_data(
                    train_x, train_y, test_x, b, context_size
                )
                with self.autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    _, src = self.model(
                        x=torch.cat([X_nni, X_eval], dim=0),
                        y=y_nni,
                        return_embeddings=True,
                    )
                # src is [context_size + n_batch, batch_size, ninp]
                res_t = src[-1, :, :].detach().to(torch.float32)
                if hasattr(self, 'norm') and self.norm is not None:
                    res_t = self.norm(res_t)
                
                batch_emb = res_t.cpu().numpy()
                embeddings.append(batch_emb)
            
            res = np.concatenate(embeddings, axis=0)
            if return_full:
                # For retrieval, we don't have a single shared "full" sequence
                # Just return query embeddings as [n_test, 1, ninp]
                return res[:, None, :]
            return res
