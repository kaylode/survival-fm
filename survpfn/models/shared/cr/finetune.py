"""survpfn.models.shared.cr.finetune — Competing-risks joint fine-tuning.

Classes
-------
BaseCRJointFinetune
    Extends ``BaseJointSurvFinetune`` and overrides four template methods
    so that the generic ``fit()`` loop works correctly for competing risks:

    * ``_compute_survival_loss``  — uses ``CRDeepHitLoss`` / ``CRCoxLoss``
    * ``_compute_baseline_hazards`` — no-op (CR has no Cox baseline hazard)
    * ``predict_survival_df``     — returns ``List[pd.DataFrame]`` in pycox
                                    convention (rows = times, cols = patients)
    * ``evaluate``                — binarises CR event codes before ``EvalSurv``

TabPFNCRPH / TabDPTCRPH / TabICLCRPH
    Thin wrappers that instantiate the right backbone and call ``_setup_cr_loss()``.
    They set ``self.model = None`` and ``self.labtrans = None`` because
    CR training does not use pycox wrappers.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from survpfn.models.shared.finetune import BaseJointSurvFinetune
from survpfn.models.shared.cr.loss import CRDeepHitLoss, CRCoxLoss


# ---------------------------------------------------------------------------
# BaseCRJointFinetune
# ---------------------------------------------------------------------------

class BaseCRJointFinetune(BaseJointSurvFinetune):
    """Competing-risks joint fine-tuning base class.

    Inherits the full ``fit()`` training loop from ``BaseJointSurvFinetune``
    (sampler, optimizer, early-stopping, etc.) and overrides only the parts
    that differ for CR:

    1. Loss computation: ``CRDeepHitLoss`` or ``CRCoxLoss``.
    2. Baseline hazard: no-op (CR models have no baseline hazard).
    3. Prediction: per-cause CIF DataFrames in pycox convention.
    4. Evaluation: binarise CR event codes before ``EvalSurv``.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_cr_loss(self) -> None:
        """Instantiate the CR loss function.  Call once in ``__init__``."""
        if self.head_type in ("deephit",):
            alpha = getattr(self, "deephit_alpha", 1.0)
            self._cr_loss_fn: nn.Module = CRDeepHitLoss(alpha=alpha)
        elif self.head_type in ("cox", "deepsurv"):
            self._cr_loss_fn = CRCoxLoss()
        else:
            raise ValueError(
                f"CR mode only supports head_type in ('deephit', 'cox', 'deepsurv'), "
                f"got '{self.head_type}'."
            )

    # ------------------------------------------------------------------
    # Template method overrides
    # ------------------------------------------------------------------

    def _compute_survival_loss(
        self,
        head_out: torch.Tensor,
        bdur: torch.Tensor,
        bdur_cont: torch.Tensor,
        bev: torch.Tensor,
        bfrac,        # unused for CR
        criterion_surv,  # unused for CR (we use _cr_loss_fn)
    ) -> torch.Tensor:
        """CR log-likelihood: DeepHit or Cox depending on ``head_type``."""
        if self.head_type in ("cox", "deepsurv"):
            # Cox uses continuous times for sorting
            return self._cr_loss_fn(head_out, bdur_cont, bev)
        # DeepHit uses discretised bin indices
        return self._cr_loss_fn(head_out, bdur, bev)

    def _compute_baseline_hazards(
        self,
        x_pt: torch.Tensor,
        dur_pt: torch.Tensor,
        ev_pt: torch.Tensor,
    ) -> None:
        """No-op — CR models have no Cox baseline hazard."""
        pass

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_survival_df(
        self, x: np.ndarray, n_ensemble: int = 0
    ) -> List[pd.DataFrame]:
        """Return per-cause CIF DataFrames in pycox convention.

        Returns
        -------
        List[pd.DataFrame]
            One DataFrame per cause.  Each DataFrame has shape
            ``(n_times, n_patients)`` with ``index = grid_times`` and
            ``columns = np.arange(N)``, matching pycox's ``EvalSurv``
            expectation.
        """
        from scipy.interpolate import interp1d

        self.net.eval()
        x_pt = self._prep.transform(x, device=self.device)
        N = x_pt.shape[0]

        def _single() -> List[pd.DataFrame]:
            with torch.no_grad():
                out = self.net(x_pt)       # (N, K, D) — joint PMF
            out_np = out.cpu().numpy()
            K = out_np.shape[1]
            D = out_np.shape[2]

            times = np.concatenate([[0.0], self._bin_times])
            t_max = float(self._bin_times.max())
            grid  = np.linspace(0.0, t_max, 1000)

            cifs: List[pd.DataFrame] = []
            for k in range(K):
                cif_disc = np.cumsum(out_np[:, k, :], axis=1)  # (N, D)
                cif_ext  = np.concatenate(
                    [np.zeros((N, 1)), cif_disc], axis=1
                )                                                # (N, D+1)
                f        = interp1d(
                    times, cif_ext, kind="linear",
                    axis=1, fill_value="extrapolate",
                )
                cif_grid = np.clip(f(grid), 0.0, 1.0)           # (N, 1000)
                # pycox convention: rows = times, cols = patients
                cifs.append(
                    pd.DataFrame(
                        cif_grid.T,
                        index=grid,
                        columns=np.arange(N),
                    )
                )
            return cifs

        if n_ensemble > 0 and hasattr(self, "_train_x"):
            ensemble: List[List[pd.DataFrame]] = []
            orig_x, orig_y = self.net._ctx_x, self.net._ctx_y
            N_tr = len(self._train_x)
            for _ in range(n_ensemble):
                idx = torch.randperm(N_tr, device=self.device)[: self.context_size]
                x_c = self._train_x[idx]
                if hasattr(self.net, "_to_padded"):
                    x_c = self.net._to_padded(x_c)
                self.net.set_context(x_c, self._train_y[idx])
                ensemble.append(_single())
            self.net.set_context(orig_x, orig_y)

            result: List[pd.DataFrame] = []
            for k in range(self.num_events):
                vals = np.mean([e[k].values for e in ensemble], axis=0)
                result.append(
                    pd.DataFrame(
                        vals,
                        index=ensemble[0][k].index,
                        columns=ensemble[0][k].columns,
                    )
                )
            return result

        return _single()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        surv_dfs: List[pd.DataFrame],
        durations_test: np.ndarray,
        events_test: np.ndarray,
        durations_train: np.ndarray,
        events_train: np.ndarray,
    ) -> dict:
        """Evaluate using cause-1 CIF; binarise CR event codes for EvalSurv.

        ``EvalSurv`` does not understand multi-class event codes.  We
        binarise: cause-1 patients are labelled 1, everyone else
        (censored *or* competing-cause event) is labelled 0.  This gives
        the correct cause-specific C-index for cause 1.
        """
        from pycox.evaluation import EvalSurv

        # Binary: cause-1 = 1, censored/competing = 0
        events_bin = (np.asarray(events_test) == 1).astype(float)

        cif_1  = surv_dfs[0]   # (n_times, n_patients)
        surv_1 = 1.0 - cif_1  # 1 − CIF_1 used as "survival" for EvalSurv

        ev = EvalSurv(
            surv_1,
            np.asarray(durations_test),
            events_bin,
            censor_surv="km",
        )
        return {"c_index": ev.concordance_td("antolini")}


# ---------------------------------------------------------------------------
# Thin backbone wrappers
# ---------------------------------------------------------------------------

class TabPFNCRPH(BaseCRJointFinetune):
    """TabPFN backbone + competing-risks head (DeepHit or Cox).

    Parameters
    ----------
    num_events     : int  — number of competing causes (default 2).
    num_durations  : int  — number of discrete time bins.
    head_type      : str  — ``"deephit"`` or ``"cox"`` (default ``"deephit"``).
    freeze_backbone: bool — keep TabPFN weights frozen (default ``True``).
    """

    def __init__(
        self,
        num_events: int = 2,
        num_durations: int = 50,
        head_type: str = "deephit",
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.float32,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        context_size: int = 1024,
        deephit_alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        # Lazy import to avoid circular imports at module load time
        from survpfn.models.tabpfn.model import TabPFNSurvModel

        self.task_type     = "cr"
        self.num_events    = num_events
        self.head_type     = head_type.lower()
        self.dtype         = dtype
        self.context_size  = context_size
        self.deephit_alpha = deephit_alpha
        self.backbone_name = "tabpfn"
        # CR models do not use pycox wrappers
        self.model         = None
        self.labtrans      = None

        self.net = TabPFNSurvModel(
            n_out=num_durations * num_events,
            head_num_nodes=head_num_nodes,
            dropout=dropout,
            freeze_tabpfn=freeze_backbone,
            dtype=dtype,
            device=self.device,
            task_type="cr",
            num_events=num_events,
            use_adapter=use_adapter,
            input_dim=input_dim,
        )
        self._setup_cr_loss()


class TabDPTCRPH(BaseCRJointFinetune):
    """TabDPT backbone + competing-risks head (DeepHit or Cox).

    Parameters
    ----------
    num_events     : int  — number of competing causes (default 2).
    num_durations  : int  — number of discrete time bins.
    head_type      : str  — ``"deephit"`` or ``"cox"`` (default ``"deephit"``).
    freeze_backbone: bool — keep TabDPT weights frozen (default ``True``).
    """

    def __init__(
        self,
        num_events: int = 2,
        num_durations: int = 50,
        head_type: str = "deephit",
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        context_size: int = 128,
        deephit_alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        from survpfn.models.tabdpt.model import TabDPTSurvModel

        self.task_type     = "cr"
        self.num_events    = num_events
        self.head_type     = head_type.lower()
        self.context_size  = context_size
        self.deephit_alpha = deephit_alpha
        self.backbone_name = "tabdpt"
        self.model         = None
        self.labtrans      = None

        self.net = TabDPTSurvModel(
            device=self.device,
            n_out=num_durations * num_events,
            head_num_nodes=head_num_nodes,
            dropout=dropout,
            task_type="cr",
            num_events=num_events,
            use_adapter=use_adapter,
            input_dim=input_dim,
            freeze_tabdpt=freeze_backbone,
        )
        self._setup_cr_loss()


class TabICLCRPH(BaseCRJointFinetune):
    """TabICL backbone + competing-risks head (DeepHit or Cox).

    Parameters
    ----------
    num_events     : int  — number of competing causes (default 2).
    num_durations  : int  — number of discrete time bins.
    head_type      : str  — ``"deephit"`` or ``"cox"`` (default ``"deephit"``).
    freeze_backbone: bool — keep TabICL weights frozen (default ``True``).
    """

    def __init__(
        self,
        num_events: int = 2,
        num_durations: int = 50,
        head_type: str = "deephit",
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        context_size: int = 512,
        deephit_alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(num_durations=num_durations, **kwargs)
        from survpfn.models.tabicl.model import TabICLSurvModel

        self.task_type     = "cr"
        self.num_events    = num_events
        self.head_type     = head_type.lower()
        self.context_size  = context_size
        self.deephit_alpha = deephit_alpha
        self.backbone_name = "tabicl"
        self.model         = None
        self.labtrans      = None

        self.net = TabICLSurvModel(
            n_out=num_durations * num_events,
            head_num_nodes=head_num_nodes,
            dropout=dropout,
            device=self.device,
            task_type="cr",
            num_events=num_events,
            use_adapter=use_adapter,
            input_dim=input_dim,
            freeze_backbone=freeze_backbone,
        ).to(self.device)
        self._setup_cr_loss()
