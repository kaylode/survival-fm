"""
src/models.py — Survival model wrappers with a consistent API.

All wrappers implement:
    fit(df_train, duration_col, event_col, **kwargs) -> self
    predict_risk(df_test)  -> np.ndarray  (higher = worse prognosis)
    predict_survival(df_test) -> (surv_probs: np.ndarray, surv_times: np.ndarray)

Models
------
CoxPHWrapper         : lifelines CoxPHFitter (interpretable baseline)
DeepSurvWrapper      : pycox CoxPH / DeepSurv (deep proportional hazards)
DeepHitWrapper       : pycox DeepHitSingle (discrete-time, competing-risk ready)
RSFWrapper           : scikit-survival RandomSurvivalForest
GBSAWrapper          : scikit-survival GradientBoostingSurvivalAnalysis
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from lifelines import CoxPHFitter
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest


# ---------------------------------------------------------------------------
# 1.  CoxPH (lifelines)
# ---------------------------------------------------------------------------

class CoxPHWrapper:
    """Wrapper around ``lifelines.CoxPHFitter`` with a unified predict API.

    Parameters
    ----------
    penalizer:
        L2 regularisation coefficient.  Lifelines default is 0.
    step_size:
        Optimisation step size for the Newton-Raphson solver.
    """

    def __init__(self, penalizer: float = 0.1, step_size: float = 0.5) -> None:
        self.penalizer = penalizer
        self.step_size = step_size
        self._model: Optional[CoxPHFitter] = None
        self._duration_col: str = ""
        self._event_col: str = ""

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "CoxPHWrapper":
        """Fit the Cox model.

        Parameters
        ----------
        df_train:
            Training DataFrame (already scaled; must include duration + event).
        duration_col:
            Name of the time column.
        event_col:
            Name of the event indicator column.

        Returns
        -------
        self
        """
        self._duration_col = duration_col
        self._event_col = event_col
        self._model = CoxPHFitter(penalizer=self.penalizer, step_size=self.step_size)
        self._model.fit(df_train, duration_col=duration_col, event_col=event_col)
        return self

    def predict_risk(self, df_test: pd.DataFrame) -> np.ndarray:
        """Return per-subject log partial hazard (higher = higher risk).

        Parameters
        ----------
        df_test:
            Test DataFrame (same columns as training, minus duration + event).
        """
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(columns=[self._duration_col, self._event_col], errors="ignore")
        return self._model.predict_log_partial_hazard(X).values

    def predict_survival(
        self, df_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted survival curves.

        Returns
        -------
        (surv_probs, surv_times)
            ``surv_probs``  : shape ``(n_test, n_times)``
            ``surv_times``  : 1-D array of time points
        """
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(columns=[self._duration_col, self._event_col], errors="ignore")
        surv_df = self._model.predict_survival_function(X)  # times × subjects
        surv_times = surv_df.index.values.astype(float)
        surv_probs = surv_df.values.T  # (n_subjects, n_times)
        return surv_probs, surv_times

    @property
    def summary(self) -> pd.DataFrame:
        """Coefficient summary table from lifelines."""
        assert self._model is not None, "Call fit() first."
        return self._model.summary

    @property
    def concordance_index(self) -> float:
        """Training-set C-index reported by lifelines."""
        assert self._model is not None, "Call fit() first."
        return self._model.concordance_index_


# ---------------------------------------------------------------------------
# 2.  DeepSurv (pycox CoxPH)
# ---------------------------------------------------------------------------

class DeepSurvWrapper:
    """Wrapper around ``pycox.models.CoxPH`` (DeepSurv).

    Parameters
    ----------
    num_nodes:
        Hidden-layer sizes for the MLP.
    dropout:
        Dropout rate.
    lr:
        Learning rate.
    batch_size:
        Mini-batch size.
    epochs:
        Number of training epochs.
    """

    def __init__(
        self,
        num_nodes: list[int] = (32, 32),
        dropout: float = 0.1,
        lr: float = 1e-2,
        batch_size: int = 128,
        epochs: int = 100,
    ) -> None:
        self.num_nodes = list(num_nodes)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self._model: Optional[CoxPH] = None
        self._duration_col: str = ""
        self._event_col: str = ""
        self._surv_df: Optional[pd.DataFrame] = None

    def _build(self, in_features: int) -> CoxPH:
        net = tt.practical.MLPVanilla(
            in_features,
            self.num_nodes,
            1,
            batch_norm=True,
            dropout=self.dropout,
            activation=nn.ReLU,
        )
        optimizer = tt.optim.AdamWR(lr=self.lr)
        return CoxPH(net, optimizer)

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
        verbose: bool = False,
    ) -> "DeepSurvWrapper":
        """Train DeepSurv on ``df_train``.

        Parameters
        ----------
        df_train:
            Scaled training data with duration and event columns.
        duration_col:
            Time column name.
        event_col:
            Event column name.
        verbose:
            Print training progress.

        Returns
        -------
        self
        """
        self._duration_col = duration_col
        self._event_col = event_col

        X = df_train.drop(columns=[duration_col, event_col]).values.astype("float32")
        T = df_train[duration_col].values.astype("float32")
        E = df_train[event_col].values.astype("float32")

        X_t = torch.tensor(X, dtype=torch.float32)
        T_t = torch.tensor(T, dtype=torch.float32)
        E_t = torch.tensor(E, dtype=torch.float32)

        self._model = self._build(X.shape[1])
        self._model.fit(X_t, (T_t, E_t), self.batch_size, self.epochs, verbose=verbose)
        self._model.compute_baseline_hazards()
        return self

    def predict_risk(self, df_test: pd.DataFrame) -> np.ndarray:
        """Return log partial hazard scores."""
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(
            columns=[self._duration_col, self._event_col], errors="ignore"
        ).values.astype("float32")
        X_t = torch.tensor(X, dtype=torch.float32)
        return self._model.predict(X_t).squeeze()

    def predict_survival(
        self, df_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return survival probability matrix and time grid.

        Returns
        -------
        (surv_probs, surv_times)
        """
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(
            columns=[self._duration_col, self._event_col], errors="ignore"
        ).values.astype("float32")
        X_t = torch.tensor(X, dtype=torch.float32)
        surv_df = self._model.predict_surv_df(X_t)
        self._surv_df = surv_df
        surv_times = surv_df.index.values.astype(float)
        surv_probs = surv_df.values.T  # (n_subjects, n_times)
        return surv_probs, surv_times

    def eval_surv(self, df_test: pd.DataFrame) -> EvalSurv:
        """Return a ``pycox.EvalSurv`` object for concordance / IBS via pycox."""
        assert self._surv_df is not None, "Call predict_survival() first."
        T = df_test[self._duration_col].values
        E = df_test[self._event_col].values
        return EvalSurv(self._surv_df, T, E, censor_surv="km")


# ---------------------------------------------------------------------------
# 3.  DeepHit (pycox DeepHitSingle)
# ---------------------------------------------------------------------------

class DeepHitWrapper:
    """Wrapper around ``pycox.models.DeepHitSingle``.

    Suitable for single-event and competing-risk settings.  For competing risks
    the model predicts the sub-distribution hazard; ``predict_risk`` returns the
    CIF at the last time point (1 − survival) which monotonically ranks subjects.

    Parameters
    ----------
    num_nodes:
        Hidden-layer sizes.
    dropout:
        Dropout rate.
    lr:
        Learning rate.
    batch_size:
        Mini-batch size.
    epochs:
        Training epochs.
    num_durations:
        Number of discrete time bins.  More bins = finer time resolution but
        slower training.
    """

    def __init__(
        self,
        num_nodes: list[int] = (64, 32),
        dropout: float = 0.1,
        lr: float = 1e-2,
        batch_size: int = 128,
        epochs: int = 50,
        num_durations: int = 100,
    ) -> None:
        self.num_nodes = list(num_nodes)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_durations = num_durations
        self._model: Optional[DeepHitSingle] = None
        self._labtrans: Optional[LabTransDiscreteTime] = None
        self._duration_col: str = ""
        self._event_col: str = ""
        self._surv_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
        verbose: bool = False,
    ) -> "DeepHitWrapper":
        """Train DeepHit on ``df_train``.

        Parameters
        ----------
        df_train:
            Scaled training data.
        duration_col:
            Time column.
        event_col:
            Event column (0 = censored, 1 / 2 = competing causes).
        verbose:
            Print progress.

        Returns
        -------
        self
        """
        self._duration_col = duration_col
        self._event_col = event_col

        X = df_train.drop(columns=[duration_col, event_col]).values.astype("float32")
        T = df_train[duration_col].values
        E = df_train[event_col].values

        self._labtrans = LabTransDiscreteTime(self.num_durations)
        y_train = self._labtrans.fit_transform(T, E)
        out_features = self._labtrans.out_features

        net = tt.practical.MLPVanilla(
            X.shape[1],
            self.num_nodes,
            out_features=out_features,
            batch_norm=True,
            dropout=self.dropout,
            activation=nn.ReLU,
        )
        optimizer = tt.optim.AdamWR(lr=self.lr)
        self._model = DeepHitSingle(net, optimizer, duration_index=self._labtrans.cuts)
        self._model.fit(X, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)
        return self

    def predict_risk(self, df_test: pd.DataFrame) -> np.ndarray:
        """Return CIF at the last observed time point as a risk score."""
        surv_probs, _ = self.predict_survival(df_test)
        # 1 - survival at last time point = cumulative incidence
        return 1.0 - surv_probs[:, -1]

    def predict_survival(
        self, df_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return survival probability matrix and time grid.

        Returns
        -------
        (surv_probs, surv_times)
        """
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(
            columns=[self._duration_col, self._event_col], errors="ignore"
        ).values.astype("float32")
        surv_df = self._model.predict_surv_df(X)
        self._surv_df = surv_df
        surv_times = surv_df.index.values.astype(float)
        surv_probs = surv_df.values.T  # (n_subjects, n_times)
        return surv_probs, surv_times

    def eval_surv(self, df_test: pd.DataFrame) -> EvalSurv:
        """Return a ``pycox.EvalSurv`` object."""
        assert self._surv_df is not None, "Call predict_survival() first."
        T = df_test[self._duration_col].values
        E = df_test[self._event_col].values
        return EvalSurv(self._surv_df, T, E, censor_surv="km")


# ---------------------------------------------------------------------------
# 4.  Random Survival Forest  (scikit-survival)
# ---------------------------------------------------------------------------

class RSFWrapper:
    """Wrapper around ``sksurv.ensemble.RandomSurvivalForest``.

    Parameters
    ----------
    n_estimators:
        Number of trees.
    min_samples_split:
        Minimum samples to split a node.
    min_samples_leaf:
        Minimum samples in a leaf.
    n_jobs:
        Number of parallel jobs (``-1`` = use all cores).
    random_state:
        Seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 15,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: Optional[RandomSurvivalForest] = None
        self._duration_col: str = ""
        self._event_col: str = ""

    def _to_sksurv_y(self, df: pd.DataFrame) -> np.ndarray:
        event = df[self._event_col].astype(bool)
        time = df[self._duration_col]
        return np.array(
            list(zip(event, time)),
            dtype=[("event", "?"), ("time", "<f8")],
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "RSFWrapper":
        """Train the Random Survival Forest.

        Parameters
        ----------
        df_train:
            Training data.
        duration_col:
            Time column.
        event_col:
            Event column.

        Returns
        -------
        self
        """
        self._duration_col = duration_col
        self._event_col = event_col
        X = df_train.drop(columns=[duration_col, event_col])
        y = self._to_sksurv_y(df_train)
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict_risk(self, df_test: pd.DataFrame) -> np.ndarray:
        """Return cumulative hazard (risk) predictions."""
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(columns=[self._duration_col, self._event_col], errors="ignore")
        return self._model.predict(X)

    def predict_survival(
        self, df_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return survival probability matrix and time grid.

        Returns
        -------
        (surv_probs, surv_times)
        """
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(columns=[self._duration_col, self._event_col], errors="ignore")
        surv_times = self._model.unique_times_
        surv_funcs = self._model.predict_survival_function(X)
        surv_probs = np.vstack([fn(surv_times) for fn in surv_funcs])
        return surv_probs, surv_times


# ---------------------------------------------------------------------------
# 5.  Gradient Boosting Survival Analysis  (scikit-survival)
# ---------------------------------------------------------------------------

class GBSAWrapper:
    """Wrapper around ``sksurv.ensemble.GradientBoostingSurvivalAnalysis``.

    Parameters
    ----------
    learning_rate:
        Boosting learning rate (shrinkage).
    n_estimators:
        Number of boosting stages.
    max_depth:
        Maximum tree depth.
    random_state:
        Seed.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Optional[GradientBoostingSurvivalAnalysis] = None
        self._duration_col: str = ""
        self._event_col: str = ""

    def _to_sksurv_y(self, df: pd.DataFrame) -> np.ndarray:
        event = df[self._event_col].astype(bool)
        time = df[self._duration_col]
        return np.array(
            list(zip(event, time)),
            dtype=[("event", "?"), ("time", "<f8")],
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "GBSAWrapper":
        """Train GBSA.

        Parameters
        ----------
        df_train:
            Training data.
        duration_col:
            Time column.
        event_col:
            Event column.

        Returns
        -------
        self
        """
        self._duration_col = duration_col
        self._event_col = event_col
        X = df_train.drop(columns=[duration_col, event_col])
        y = self._to_sksurv_y(df_train)
        self._model = GradientBoostingSurvivalAnalysis(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict_risk(self, df_test: pd.DataFrame) -> np.ndarray:
        """Return cumulative hazard (risk) scores."""
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(columns=[self._duration_col, self._event_col], errors="ignore")
        return self._model.predict(X)

    def predict_survival(
        self, df_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return survival probability matrix and time grid.

        Returns
        -------
        (surv_probs, surv_times)
        """
        assert self._model is not None, "Call fit() first."
        X = df_test.drop(columns=[self._duration_col, self._event_col], errors="ignore")
        surv_times = self._model.unique_times_
        surv_funcs = self._model.predict_survival_function(X)
        surv_probs = np.vstack([fn(surv_times) for fn in surv_funcs])
        return surv_probs, surv_times
