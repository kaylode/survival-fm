from __future__ import annotations

import copy
import warnings
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


import torchtuples as tt
from pycox.models import CoxPH, PCHazard, MTLR, DeepHitSingle
from pycox.models.loss import CoxPHLoss, DeepHitSingleLoss, NLLPCHazardLoss, NLLMTLRLoss
from pycox.models.data import pair_rank_mat

from survpfn.models.shared.preprocessing import (
    FMDataPrep,
    SurvivalTimeBinEncoder,
    expand_survival_data,
    FMTargetPrep
)
from survpfn.models.shared.loss import (
    compute_deephit_cr_loss,
    compute_deephit_cr_loss_v2,
    compute_cox_cr_loss
)


# ---------------------------------------------------------------------------
# EventBalancedBatchSampler
# ---------------------------------------------------------------------------

class EventBalancedBatchSampler:
    """Generates batch index arrays guaranteeing ≥ ``min_events`` event samples
    per batch.

    Root cause this fixes
    ---------------------
    CoxPH partial-likelihood is undefined (→ NaN loss) when a mini-batch
    contains *zero* events.  On low-event-rate datasets such as MIMIC-IV
    (event rate ≈ 8–15 %) this happens frequently with standard random
    shuffling, especially for larger batch sizes.

    Strategy
    --------
    Training indices are split into an *event pool* and a *censored pool*.
    Each batch is assembled from:

    * exactly ``min_events`` samples drawn from the event pool (without
      replacement within an epoch; the pool is re-shuffled when exhausted), and
    * ``batch_size − min_events`` samples drawn from a global permutation of
      **all** training indices (so events can also appear in the fill portion).

    The assembled batch is shuffled before yielding so events are not clustered
    at the front — this matters for CoxPH which relies on the rank order of
    event times within the batch.

    Parameters
    ----------
    events : array-like, shape (N,)
        Event indicator for each training sample (0 = censored, >0 = event).
        Competing-risks codes (2, 3, …) are treated as events.
    batch_size : int
        Target batch size.
    min_events : int, optional
        Minimum number of event samples per batch.  Defaults to
        ``max(1, batch_size // 8)``.  Clamped to
        ``min(min_events, n_events_in_dataset, batch_size // 2)``.
    seed : int, optional
        NumPy seed for reproducible sampling.
    """

    def __init__(
        self,
        events: np.ndarray,
        batch_size: int,
        min_events: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        events = np.asarray(events)
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._n   = len(events)

        self._event_idx  = np.where(events > 0)[0]
        self._censor_idx = np.where(events == 0)[0]

        if len(self._event_idx) == 0:
            raise ValueError(
                "Training fold has no observed events — "
                "survival model cannot be trained."
            )

        # Resolve and clamp min_events
        default_min = max(1, batch_size // 8)
        raw = min_events if min_events is not None else default_min
        self.min_events = int(min(raw, len(self._event_idx), max(1, batch_size // 2)))
        self._fill_size = batch_size - self.min_events

    # ------------------------------------------------------------------
    def __iter__(self):
        """Yield one shuffled LongTensor of indices per batch, for one epoch."""
        # Shuffle event pool at epoch start; recycle if exhausted mid-epoch
        e_perm = self._rng.permutation(self._event_idx)
        e_ptr  = 0

        # Shuffle ALL indices for the fill portion; recycle independently
        all_perm = self._rng.permutation(self._n)
        f_ptr    = 0

        while e_ptr < len(e_perm):
            # ── Guaranteed event slice ─────────────────────────────────────
            e_end = e_ptr + self.min_events
            if e_end <= len(e_perm):
                e_batch = e_perm[e_ptr:e_end]
                e_ptr   = e_end
            else:
                # Fewer events remain than min_events → wrap the event pool
                tail    = e_perm[e_ptr:]
                e_perm  = self._rng.permutation(self._event_idx)
                need    = self.min_events - len(tail)
                e_batch = np.concatenate([tail, e_perm[:need]])
                e_ptr   = need

            # ── Fill slice (drawn from all indices) ────────────────────────
            if self._fill_size > 0:
                chunks, need = [], self._fill_size
                while need > 0:
                    avail = all_perm[f_ptr:]
                    take  = min(need, len(avail))
                    chunks.append(avail[:take])
                    need  -= take
                    f_ptr += take
                    if f_ptr >= len(all_perm):   # recycle
                        all_perm = self._rng.permutation(self._n)
                        f_ptr    = 0
                fill_batch = np.concatenate(chunks)
                batch_np   = np.concatenate([e_batch, fill_batch])
            else:
                batch_np = e_batch

            # Shuffle within batch so events are not always at front
            self._rng.shuffle(batch_np)
            yield torch.from_numpy(batch_np.astype(np.int64))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        return max(1, len(self._event_idx) // self.min_events)


# ---------------------------------------------------------------------------
# Base Joint Survival nn.Module
# ---------------------------------------------------------------------------

class BaseBackboneSurvModel(nn.Module):
    """Base nn.Module for models combining a backbone with a survival head.
    
    Shared logic for:
    1. Initializing survival/competing-risks heads.
    2. Initializing input adapters.
    3. Managing context and PCA projections.
    4. Weight initialization.
    """
    def __init__(
        self,
        ninp: int,
        n_out: int,
        head_num_nodes: List[int] = [128, 64],
        dropout: float = 0.2,
        task_type: str = "sr",
        num_events: int = 1,
        use_adapter: bool = False,
        input_dim: Optional[int] = None,
        batch_norm: bool = False,
        adapter_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.ninp = ninp
        self.out_features = n_out
        self.task_type = task_type
        self.num_events = num_events
        self.adapter_output_dim = adapter_output_dim

        # --- (1) Input Adapter ---
        if use_adapter:
            if input_dim is None or adapter_output_dim is None:
                raise ValueError("input_dim and adapter_output_dim must be provided if use_adapter is True")
            self.input_adapter = nn.Sequential(
                nn.Linear(input_dim, adapter_output_dim),
                nn.ReLU(),
                nn.Linear(adapter_output_dim, adapter_output_dim)
            )
        else:
            self.input_adapter = None

        # --- (2) Heads ---
        self.norm = nn.LayerNorm(self.ninp)
        
        if task_type == "cr":
            hidden_dim = head_num_nodes[0] if len(head_num_nodes) > 0 else 128
            self.cs_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.ninp, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_out // num_events)
                ) for _ in range(num_events)
            ])
            self.survival_head = None
        else:
            # Multi-layer survival head
            layers = []
            prev_dim = self.ninp
            for h_dim in head_num_nodes:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, n_out, bias=False))
            self.survival_head = nn.Sequential(*layers)

        self._ctx_x: Optional[torch.Tensor] = None
        self._ctx_y: Optional[torch.Tensor] = None

    def _move_head_to_last(self):
        """Move survival head to the end of _modules so pycox finds its out_features."""
        if self.task_type == "cr":
            temp = self.cs_heads
            del self.cs_heads
            self.cs_heads = temp
        else:
            temp = self.survival_head
            del self.survival_head
            self.survival_head = temp
        self._pca_v: Optional[torch.Tensor] = None

    def set_context(self, x_context: torch.Tensor, y_context: torch.Tensor) -> None:
        self._ctx_x = x_context
        self._ctx_y = y_context

    def set_pca(self, V: Optional[torch.Tensor]) -> None:
        self._pca_v = V
        # to device
        if self._pca_v is not None:
            self._pca_v = self._pca_v.to(self.device)

    def _apply_pca_and_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        if self._pca_v is not None:
            # Only apply if input dimension matches the PCA matrix input dimension (rows)
            # This avoids double-projection if FMDataPrep already applied it.
            if x.shape[-1] == self._pca_v.shape[0]:
                x = x @ self._pca_v
        return x


# ---------------------------------------------------------------------------
# BaseSurvFinetune
# ---------------------------------------------------------------------------

class BaseSurvExpandedFinetune:
    """Base class for survival finetuning via temporal expansion.
    
    Shared logic for:
    1. Dataset expansion (survival -> binary classification)
    2. Monotone survival path reconstruction
    3. Standard evaluation
    """

    def _expand_data_numpy(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        bin_times: np.ndarray,
        bin_feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return expand_survival_data(X, T, E, bin_times, bin_feats)

    def predict_survival_df(self, x: np.ndarray, n_ensemble: int = 0) -> pd.DataFrame:
        """Return survival probability DataFrame (rows=times, cols=subjects)."""
        x_scaled = self._prep.transform(x)
        n_test = len(x_scaled)
        K = len(self.bin_times)

        def _get_surv_matrix():
            surv_m = np.zeros((n_test, K), dtype=np.float32)
            for k in range(K):
                f_k = np.repeat(self.bin_feats[k : k + 1], n_test, axis=0)
                x_bin = np.concatenate([x_scaled, f_k], axis=1)
                probs = self._predict_proba_internal(x_bin)
                surv_m[:, k] = probs[:, 0]  # P(survived)
            return surv_m

        if n_ensemble > 0 and hasattr(self, "_train_x") and hasattr(self.net, "set_context"):
            matrices = []
            orig_ctx_x, orig_ctx_y = self.net._ctx_x, self.net._ctx_y
            N_tr = len(self._train_x)
            for _ in range(n_ensemble):
                ctx_idx = torch.randperm(N_tr)[:self.context_size]
                x_ctx = self._train_x[ctx_idx].to(self.device)
                if hasattr(self.net, "_to_padded"): x_ctx = self.net._to_padded(x_ctx)
                y_ctx = self._train_y[ctx_idx].float().to(self.device)
                self.net.set_context(x_ctx, y_ctx)
                matrices.append(_get_surv_matrix())
            self.net.set_context(orig_ctx_x, orig_ctx_y)
            surv_matrix = np.mean(matrices, axis=0)
        else:
            surv_matrix = _get_surv_matrix()

        # ── Interpolation (Unique times only) ──
        times = np.concatenate([[0], self.bin_times])
        surv = np.concatenate([np.ones((n_test, 1)), surv_matrix], axis=1)

        # Remove duplicate zeros if present
        times, unique_idx = np.unique(times, return_index=True)
        surv = surv[:, unique_idx]

        grid_times = np.linspace(0, self.bin_times.max(), 1000)
        from scipy.interpolate import interp1d
        f = interp1d(times, surv, kind='linear', axis=1, fill_value="extrapolate")
        surv_matrix_interp = np.clip(f(grid_times), 0.0, 1.0)

        return pd.DataFrame(
            surv_matrix_interp.T,
            index=grid_times,
            columns=np.arange(n_test),
        )

    def _predict_proba_internal(self, x_bin: np.ndarray) -> np.ndarray:
        """Internal helper to be implemented by child. Should return (N, 2)."""
        raise NotImplementedError("Child must implement _predict_proba_internal")

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        val_split: float = 0.2,
        epochs: int = None,
        batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ) -> "BaseSurvExpandedFinetune":
        """Generic training loop for Expanded Survival Models (binary classification)."""
        # ── 0. Validation Split ──
        if val_data is None and val_split > 0:
            x, x_val, durations, durations_val, events, events_val = train_test_split(
                x, durations, events, test_size=val_split, random_state=kwargs.get("random_state", 42),
                stratify=events
            )
            val_data = (x_val, durations_val, events_val)


        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        # ── 1. Time bins ──
        self.encoder = SurvivalTimeBinEncoder(n_bins=self.num_durations)
        self.encoder.fit(durations, events)
        self.bin_times, self.bin_feats = self.encoder.bin_times, self.encoder.bin_feats

        # ── 2. Preprocessing ──
        self._prep = FMDataPrep()
        pca_capacity = getattr(self.net, "max_features", None)
        if pca_capacity:
             pca_capacity -= SurvivalTimeBinEncoder.N_TIME_FEATURES
        x_scaled = self._prep.fit_transform(x, max_features=pca_capacity)

        # ── 3. Expansion ──
        X_exp, y_exp = self._expand_data_numpy(x_scaled, durations, events, self.bin_times, self.bin_feats)
        M = len(X_exp)
        X_pt, y_pt = torch.from_numpy(X_exp).float(), torch.from_numpy(y_exp).long()

        if val_data is not None:
             vx, vd, ve = val_data
             vx_scaled = self._prep.transform(vx)
             VX_exp, vy_exp = self._expand_data_numpy(vx_scaled, vd, ve, self.bin_times, self.bin_feats)
             VX_pt, vy_pt = torch.from_numpy(VX_exp).float(), torch.from_numpy(vy_exp).long()
        else:
             VX_pt, vy_pt = None, None

        # ── 4. Trainer setup ──
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.learning_rate * 0.01)

        # ── 5. Main loop ──
        best_loss, best_state, patience_counter = float('inf'), None, 0
        patience = kwargs.get("patience", getattr(self, "patience", 5))
        current_bs, success = batch_size, False

        while not success:
            try:
                # Balanced sampler on the EXPANDED dataset (y_exp labels: 0=survived,
                # >0=event-at-bin-k).  Guarantees each training batch sees at least
                # min_events event rows, preventing gradient collapse toward all-survived
                # on low-event-rate datasets (e.g. MIMIC-IV ~10 % event rate).
                _train_sampler = EventBalancedBatchSampler(
                    events=y_exp,           # numpy, shape (M,), from temporal expansion
                    batch_size=current_bs,
                    seed=kwargs.get("random_state", 42),
                )

                self.net.train()
                for epoch in range(epochs):
                    epoch_loss, n_batches = 0.0, 0
                    self.net.train()    # re-assert after validation eval() each epoch

                    for batch_idx_cpu in _train_sampler:
                        batch_idx = batch_idx_cpu  # CPU tensor; index CPU X_pt/y_pt
                        bx, by = X_pt[batch_idx].to(self.device), y_pt[batch_idx].to(self.device)

                        # Context shopping
                        context_mask = torch.ones(M, dtype=torch.bool)
                        context_mask[batch_idx] = False
                        pool = context_mask.nonzero(as_tuple=True)[0]
                        n_ctx = min(self.context_size, len(pool))
                        ctx_idx = pool[torch.randperm(len(pool))[:n_ctx]]

                        x_ctx_raw, y_ctx = X_pt[ctx_idx].to(self.device), y_pt[ctx_idx].float().to(self.device)
                        x_ctx = self.net._to_padded(x_ctx_raw) if hasattr(self.net, "_to_padded") else x_ctx_raw

                        optimizer.zero_grad()
                        cls_logits = self.net(bx, x_context=x_ctx, y_context=y_ctx, return_backbone_logits=False)
                        loss = criterion(cls_logits, by)

                        if not torch.isfinite(loss):
                            warnings.warn(
                                f"[{self.backbone_name}/Expanded] epoch {epoch}: "
                                f"non-finite train loss ({loss.item():.4g}), skipping batch.",
                                stacklevel=2,
                            )
                            continue

                        loss.backward()

                        if all(torch.isfinite(p.grad).all() for p in self.net.parameters() if p.grad is not None):
                            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                            optimizer.step()
                        optimizer.zero_grad()
                        epoch_loss += loss.item()
                        n_batches += 1

                    scheduler.step()
                    avg_loss = epoch_loss / max(1, n_batches)

                    # ── Validation ──
                    if VX_pt is not None:
                        self.net.eval()
                        val_loss, n_val_batches = 0.0, 0
                        # Balanced sampler on expanded validation set — same reason as
                        # training: avoids all-censored batches and the stale NaN-after-
                        # accumulate bug (old code added to val_loss before checking).
                        _val_sampler_exp = EventBalancedBatchSampler(
                            events=vy_exp,          # numpy, shape (M_val,)
                            batch_size=batch_size,
                            seed=kwargs.get("random_state", 42) + epoch,
                        )
                        with torch.no_grad():
                            for vbidx_cpu in _val_sampler_exp:
                                vbx = VX_pt[vbidx_cpu].to(self.device)
                                vby = vy_pt[vbidx_cpu].to(self.device)
                                # Fresh context from TRAINING set for validation
                                vctx_idx = torch.randperm(M)[:min(self.context_size, M)]
                                vx_ctx_raw = X_pt[vctx_idx].to(self.device)
                                vy_ctx_raw = y_pt[vctx_idx].float().to(self.device)
                                vx_ctx = self.net._to_padded(vx_ctx_raw) if hasattr(self.net, "_to_padded") else vx_ctx_raw
                                vlogits = self.net(vbx, x_context=vx_ctx, y_context=vy_ctx_raw)
                                vl = criterion(vlogits, vby)

                                # Check BEFORE accumulating — old code added NaN to
                                # val_loss first, permanently poisoning the running total.
                                if not torch.isfinite(vl):
                                    warnings.warn(
                                        f"[{self.backbone_name}/Expanded] epoch {epoch}: "
                                        f"non-finite val loss ({vl.item():.4g}), skipping batch.",
                                        stacklevel=2,
                                    )
                                    continue
                                val_loss += vl.item()
                                n_val_batches += 1

                        if n_val_batches == 0:
                            warnings.warn(
                                f"[{self.backbone_name}/Expanded] epoch {epoch}: "
                                "ALL validation batches non-finite; using train loss for early stopping.",
                                stacklevel=2,
                            )
                            avg_val_loss = avg_loss
                        else:
                            avg_val_loss = val_loss / n_val_batches
                        stop_loss = avg_val_loss
                    else:
                        stop_loss = avg_loss

                    

                    if verbose:
                        msg = f"  [{self.backbone_name}/Expanded] epoch {epoch:3d}/{epochs} loss={avg_loss:.4f}"
                        if VX_pt is not None: msg += f" val_loss={avg_val_loss:.4f}"
                        print(msg, flush=True)

                    if stop_loss < best_loss:
                        best_loss, best_state, patience_counter = stop_loss, copy.deepcopy(self.net.state_dict()), 0
                    elif (patience_counter := patience_counter + 1) >= patience:
                        if verbose: print(f"  Early stopping at epoch {epoch}", flush=True)
                        break
                if best_state: self.net.load_state_dict(best_state)
                success = True
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and current_bs > 1:
                    torch.cuda.empty_cache(); current_bs //= 2
                    print(f"  CUDA OOM! Reducing batch_size to {current_bs}", flush=True)
                else: raise e

        # Finalize context
        n_ctx_final = min(self.context_size, M)
        ctx_idx = torch.randperm(M)[:n_ctx_final]
        with torch.no_grad():
            x_ctx_f = X_pt[ctx_idx].to(self.device)
            if hasattr(self.net, "_to_padded"): x_ctx_f = self.net._to_padded(x_ctx_f)
            self.net.set_context(x_context=x_ctx_f.detach(), y_context=y_pt[ctx_idx].float().to(self.device).detach())
        
        self._train_x = X_pt
        self._train_y = y_pt

        return self

    @staticmethod
    def evaluate(
        surv_df: pd.DataFrame,
        durations_test: np.ndarray,
        events_test: np.ndarray,
        durations_train: np.ndarray,
        events_train: np.ndarray,
    ) -> dict:
        """Compute survival evaluation metrics via pycox EvalSurv."""
        from pycox.evaluation import EvalSurv

        ev = EvalSurv(
            surv_df,
            durations_test,
            events_test,
            censor_surv="km",
        )
        c_index = ev.concordance_td("antolini")
        time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
        ibs = ev.integrated_brier_score(time_grid)

        return {"c_index": c_index, "ibs": ibs}


# ---------------------------------------------------------------------------
# BaseJointSurvFinetune (Strategy 2: Backbone + Survival Head)
# ---------------------------------------------------------------------------

class BaseJointSurvFinetune:
    """Base class for joint survival finetuning (Backbone + Pycox Head).
    
    Shared logic for:
    1. Pycox model initialization (Cox, DeepHit, PCHazard, MTLR)
    2. Survival loss functions
    3. Prediction interpolation
    """

    def _init_pycox_model(
        self,
        head_type: str,
        num_durations: int,
        learning_rate: float,
        net: nn.Module,
    ) -> tuple[Union[CoxPH, DeepHitSingle, PCHazard, MTLR], Optional[object]]:
        """Initialize Pycox wrapper and label transformer."""
        optimizer = tt.optim.Adam(lr=learning_rate)
        
        if head_type in ("cox", "deepsurv"):
            model = CoxPH(net, optimizer)
            labtrans = None
        elif head_type == "deephit":
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
            alpha = getattr(self, "deephit_alpha", 0.2)
            sigma = getattr(self, "deephit_sigma", 0.1)
            model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts, alpha=alpha, sigma=sigma)
        elif head_type == "pchazard":
            try:
                labtrans = PCHazard.label_transform(num_durations, scheme='quantiles')
            except TypeError:
                labtrans = PCHazard.label_transform(num_durations)
            model = PCHazard(net, optimizer, duration_index=labtrans.cuts)
        elif head_type == "mtlr":
            try:
                labtrans = MTLR.label_transform(num_durations, scheme='quantiles')
            except TypeError:
                labtrans = MTLR.label_transform(num_durations)
            model = MTLR(net, optimizer, duration_index=labtrans.cuts)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
            
        return model, labtrans

    def _get_criterion_surv(self, head_type: str) -> nn.Module:
        """Map head type to Pycox loss function."""
        if head_type in ("cox", "deepsurv"):
            return CoxPHLoss()
        elif head_type == "deephit":
            alpha = getattr(self, "deephit_alpha", 0.2)
            sigma = getattr(self, "deephit_sigma", 0.1)
            return DeepHitSingleLoss(alpha, sigma)
        elif head_type == "pchazard":
            return NLLPCHazardLoss()
        elif head_type == "mtlr":
            return NLLMTLRLoss()
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        val_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
        **kwargs
    ) -> "BaseJointSurvFinetune":
        """Generic joint training loop for Backbone + Survival Head."""
        # ── 0. Validation Split ──
        if val_data is None and val_split > 0:
            x, x_val, durations, durations_val, events, events_val = train_test_split(
                x, durations, events, test_size=val_split, random_state=kwargs.get("random_state", 42),
                stratify=events
            )
            val_data = (x_val, durations_val, events_val)

        N = len(x)
        
        # ── 1. Feature preprocessing ──
        self._prep = FMDataPrep()
        
        # PCA only used if NO adapter is present
        pca_capacity = None
        if not getattr(self.net, "input_adapter", None):
            pca_capacity = getattr(self.net, "max_features", None)
            if pca_capacity is None:
                pca_capacity = getattr(self.net, "num_expected_features", None)
            
        x_pt = self._prep.fit_transform(x, device=self.device, max_features=pca_capacity)
        if hasattr(self.net, "set_pca"):
            self.net.set_pca(self._prep.pca_V)

        self.net.to(self.device) 

        # ── 2. Target preparation ──
        self._target_prep = FMTargetPrep(
            task_type=self.task_type,
            head_type=self.head_type,
            num_durations=self.num_durations,
            labtrans=self.labtrans
        )
        tgt, _ = self._target_prep.fit_transform(durations, events, device=self.device)
        dur_pt, dur_pt_cont, ev_pt = tgt.dur_disc, tgt.dur_cont, tgt.events
        frac_pt = tgt.interval_frac
        if tgt.bin_times is not None:
            self._bin_times = tgt.bin_times
        
        # Ensure model's duration_index is updated (for prediction)
        if self.labtrans is not None and hasattr(self.model, 'duration_index'):
             cuts = self.labtrans.cuts
             expected_len = self.num_durations + (1 if self.head_type == "pchazard" else 0)
             if len(cuts) != expected_len:
                  # Interpolate cuts to match the expected number of time points to avoid pandas shape mismatch
                  new_cuts = np.interp(
                      np.linspace(0, len(cuts)-1, expected_len),
                      np.arange(len(cuts)),
                      cuts
                  )
                  self.model.duration_index = new_cuts
             else:
                  self.model.duration_index = cuts

        if val_data is not None:
             vx, vd, ve = val_data
             vx_pt_val = self._prep.transform(vx, device=self.device)
             vtgt = self._target_prep.transform(vd, ve, device=self.device)
             vdur_pt_val, vdur_pt_cont_val, vev_pt_val = vtgt.dur_disc, vtgt.dur_cont, vtgt.events
             vfrac_pt_val = vtgt.interval_frac
             vy_bin_pt_val = torch.from_numpy((ve > 0).astype(np.float32)).to(self.device).long() if self.backbone_name == "tabicl" else \
                             torch.from_numpy((ve > 0).astype(np.float32)).to(self.device)
        else:
             vx_pt_val = None

        # Binary event indicator for auxiliary classification loss
        y_bin_pt = torch.from_numpy((events > 0).astype(np.float32)).to(self.device)
        if self.backbone_name == "tabicl":
             y_bin_pt = y_bin_pt.long()

        # ── 3. Loss & Optimizer ──
        criterion_surv = self._get_criterion_surv(self.head_type)
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = self.model.optimizer if self.model else torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # ── 4. Main loop ──
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        patience = kwargs.get("patience", 5)
        
        current_bs = batch_size
        success = False

        while not success:
            try:
                # Build the balanced sampler once per while-attempt so that any
                # OOM-triggered batch-size reduction is reflected in a fresh sampler.
                _batch_sampler = EventBalancedBatchSampler(
                    events=events,          # original numpy array, pre-transform
                    batch_size=current_bs,
                    seed=kwargs.get("random_state", 42),
                )
                if verbose:
                    print(
                        f"  [Sampler] batch_size={current_bs}, "
                        f"min_events_per_batch={_batch_sampler.min_events}, "
                        f"n_event={len(_batch_sampler._event_idx)}/{N}",
                        flush=True,
                    )

                for epoch in range(epochs):
                    self.net.train()
                    epoch_loss = 0.0
                    n_batches  = 0

                    for batch_idx_cpu in _batch_sampler:
                        batch_idx = batch_idx_cpu.to(self.device)
                        bx, bdur, bev = x_pt[batch_idx], dur_pt[batch_idx], ev_pt[batch_idx]
                        bdur_cont, bfrac = dur_pt_cont[batch_idx], (frac_pt[batch_idx] if frac_pt is not None else None)

                        # Random context shopping
                        context_mask = torch.ones(N, dtype=torch.bool, device=self.device)
                        context_mask[batch_idx] = False
                        pool = context_mask.nonzero(as_tuple=True)[0]
                        n_ctx = min(self.context_size, len(pool))
                        ctx_idx = pool[torch.randperm(len(pool), device=self.device)[:n_ctx]]

                        x_ctx = x_pt[ctx_idx]
                        if hasattr(self.net, "_to_padded"): # For TabDPT
                             x_ctx = self.net._to_padded(x_ctx)
                        y_ctx = y_bin_pt[ctx_idx]
                        by_bin = y_bin_pt[batch_idx]

                        optimizer.zero_grad()
                        head_out, cls_logits = self.net(bx, x_context=x_ctx, y_context=y_ctx, return_logits=True)

                        # Survival Loss
                        if self.task_type == "cr":
                            if self.cr_loss_type == "deephit":
                                loss = compute_deephit_cr_loss(head_out, bdur_cont, bev, bdur, self.num_events, self.num_durations, self.device)
                            elif self.cr_loss_type == "deephit_v2":
                                loss = compute_deephit_cr_loss_v2(head_out, bdur_cont, bev, bdur, self.num_events, self.num_durations, self.device)
                            elif self.cr_loss_type == "cox":
                                loss = compute_cox_cr_loss(head_out, bdur_cont, bev, self.num_events, self.device)
                        elif self.head_type == "deephit":
                            # Use continuous durations for ranking to avoid ties
                            rank_mat = torch.from_numpy(pair_rank_mat(bdur_cont.cpu().numpy(), bev.cpu().numpy())).float().to(self.device)
                            loss = criterion_surv(head_out, bdur, bev, rank_mat)
                        elif self.head_type == "pchazard":
                            loss = criterion_surv(head_out, bdur, bev, bfrac)
                        else:
                            loss = criterion_surv(head_out, bdur, bev)

                        # Joint loss
                        total_loss = loss #+ self.alpha * criterion_cls(cls_logits, by_bin.long())
                        
                        if torch.isnan(total_loss):
                            print(f"NaN loss at epoch {epoch}", flush=True)
                            continue
                        
                        total_loss.backward()
                        
                        # Finite grad check
                        if all(torch.isfinite(p.grad).all() for p in self.net.parameters() if p.grad is not None):
                            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                            optimizer.step()
                        else:
                            optimizer.zero_grad()

                        epoch_loss += total_loss.item()
                        n_batches += 1

                    avg_loss = epoch_loss / max(1, n_batches)

                    # ── Validation ──
                    if vx_pt_val is not None:
                        self.net.eval()
                        val_loss, n_val_batches = 0.0, 0
                        # Use the same balanced sampler so validation batches are
                        # also guaranteed to contain events — sequential slicing can
                        # produce all-censored tail batches on low-event-rate datasets
                        # (e.g. MIMIC-IV ~10 % event rate) causing NaN CoxPH loss.
                        _val_sampler = EventBalancedBatchSampler(
                            events=ve,              # raw numpy from val_data
                            batch_size=batch_size,
                            seed=kwargs.get("random_state", 42) + epoch,  # vary per epoch
                        )
                        with torch.no_grad():
                            for vbidx_cpu in _val_sampler:
                                vbidx = vbidx_cpu.to(self.device)
                                vbx      = vx_pt_val[vbidx]
                                vbdur    = vdur_pt_val[vbidx]
                                vbev     = vev_pt_val[vbidx]
                                vbdur_cont = vdur_pt_cont_val[vbidx]
                                vbfrac   = vfrac_pt_val[vbidx] if vfrac_pt_val is not None else None

                                # Context always drawn from TRAINING set
                                vctx_idx = torch.randperm(N, device=self.device)[:min(self.context_size, N)]
                                vx_ctx = x_pt[vctx_idx]
                                if hasattr(self.net, "_to_padded"): vx_ctx = self.net._to_padded(vx_ctx)
                                vy_ctx = y_bin_pt[vctx_idx]

                                vhead_out, _ = self.net(vbx, x_context=vx_ctx, y_context=vy_ctx, return_logits=True)

                                if self.task_type == "cr":
                                    if self.cr_loss_type == "deephit":
                                        vl = compute_deephit_cr_loss(vhead_out, vbdur_cont, vbev, vbdur, self.num_events, self.num_durations, self.device)
                                    elif self.cr_loss_type == "deephit_v2":
                                        vl = compute_deephit_cr_loss_v2(vhead_out, vbdur_cont, vbev, vbdur, self.num_events, self.num_durations, self.device)
                                    elif self.cr_loss_type == "cox":
                                        vl = compute_cox_cr_loss(vhead_out, vbdur_cont, vbev, self.num_events, self.device)
                                elif self.head_type == "deephit":
                                    vrank_mat = torch.from_numpy(pair_rank_mat(vbdur_cont.cpu().numpy(), vbev.cpu().numpy())).float().to(self.device)
                                    vl = criterion_surv(vhead_out, vbdur, vbev, vrank_mat)
                                elif self.head_type == "pchazard":
                                    vl = criterion_surv(vhead_out, vbdur, vbev, vbfrac)
                                else:
                                    vl = criterion_surv(vhead_out, vbdur, vbev)

                                # Skip NaN batches rather than poisoning the running
                                # total — log once so the user can track frequency.
                                if not torch.isfinite(vl):
                                    warnings.warn(
                                        f"[{self.backbone_name}/{self.head_type}] "
                                        f"epoch {epoch}: non-finite val loss "
                                        f"({vl.item():.4g}), skipping batch.",
                                        stacklevel=2,
                                    )
                                    continue

                                val_loss += vl.item()
                                n_val_batches += 1

                        if n_val_batches == 0:
                            # Entire validation set produced NaN — fall back to
                            # train loss for early-stopping so training continues.
                            warnings.warn(
                                f"[{self.backbone_name}/{self.head_type}] "
                                f"epoch {epoch}: ALL validation batches were NaN; "
                                "using train loss for early stopping this epoch.",
                                stacklevel=2,
                            )
                            avg_val_loss = avg_loss
                        else:
                            avg_val_loss = val_loss / n_val_batches
                        stop_loss = avg_val_loss
                    else:
                        stop_loss = avg_loss

                    if verbose:
                        msg = f"  [{self.backbone_name}/{self.head_type}] epoch {epoch:3d}/{epochs} loss={avg_loss:.4f}"
                        if vx_pt_val is not None: msg += f" val_loss={avg_val_loss:.4f}"
                        print(msg, flush=True)

                    if stop_loss < best_loss:
                        best_loss, best_state, patience_counter = stop_loss, copy.deepcopy(self.net.state_dict()), 0
                    elif (patience_counter := patience_counter + 1) >= patience:
                        if verbose: print(f"  Early stopping at epoch {epoch}", flush=True)
                        break
                    
                    if torch.isnan(total_loss):
                        print("  NaN loss detected!", flush=True)
                        continue

                if best_state: self.net.load_state_dict(best_state)
                success = True
            except RuntimeError as e:
                # OOM recovery only makes sense if we have a backbone.
                # If static embeddings OOM, we just fail or reduce batch size.
                if "CUDA out of memory" in str(e) and current_bs > 1:
                    torch.cuda.empty_cache()
                    current_bs //= 2
                    if verbose: print(f"  CUDA OOM! Reducing batch_size to {current_bs}", flush=True)
                else: raise e

        # Finalize
        ctx_size = min(self.context_size, N)
        ctx_idx = torch.randperm(N)[:ctx_size]
        with torch.no_grad():
            x_ctx_final = x_pt[ctx_idx]
            if hasattr(self.net, "_to_padded"): x_ctx_final = self.net._to_padded(x_ctx_final)
            self.net.set_context(x_context=x_ctx_final.detach(), y_context=y_bin_pt[ctx_idx].detach())

        self._train_x = x_pt
        self._train_y = y_bin_pt

        if self.head_type in ("cox", "deepsurv"):
            self.net.eval()
            with torch.no_grad():
                self.model.compute_baseline_hazards(input=x_pt, target=(dur_pt, ev_pt))

        return self

    def predict_survival_df(self, x: np.ndarray, n_ensemble: int = 0) -> pd.DataFrame | list[pd.DataFrame]:
        """Generic survival prediction for joint models."""
        self.net.eval()
        x_pt = self._prep.transform(x, device=self.device)

        def _predict_single():
            with torch.no_grad():
                if self.task_type == "cr":
                    out = self.net(x_pt) # (N, num_events, num_durations)
                    grid_times = np.linspace(0, self._bin_times.max(), 1000)
                    from scipy.interpolate import interp1d
                    
                    cifs = []
                    times = np.concatenate([[0], self._bin_times])
                    for k in range(self.num_events):
                        # Calculate discrete CIF for cause k
                        cif_k_discrete = torch.cumsum(out[:, k, :], dim=1).cpu().numpy()
                        cif_k_ext = np.concatenate([np.zeros((len(x_pt), 1)), cif_k_discrete], axis=1)
                        
                        # Interpolate to 1000 points
                        f = interp1d(times, cif_k_ext, kind='linear', axis=1, fill_value="extrapolate")
                        cif_k_interp = np.clip(f(grid_times), 0.0, 1.0)
                        
                        cifs.append(pd.DataFrame(cif_k_interp.T, index=grid_times))
                    return cifs
                if hasattr(self.model, "interpolate"):
                    # Smooth evaluation for discrete models to break ties in risk scores during evaluation
                    return self.model.interpolate(10).predict_surv_df(x_pt)
                return self.model.predict_surv_df(x_pt)

        if n_ensemble > 0 and hasattr(self, "_train_x"):
            ensemble_results = []
            orig_ctx_x, orig_ctx_y = self.net._ctx_x, self.net._ctx_y
            N_tr = len(self._train_x)

            for _ in range(n_ensemble):
                ctx_idx = torch.randperm(N_tr, device=self.device)[:self.context_size]
                x_ctx = self._train_x[ctx_idx]
                if hasattr(self.net, "_to_padded"): x_ctx = self.net._to_padded(x_ctx)
                y_ctx = self._train_y[ctx_idx]
                self.net.set_context(x_context=x_ctx, y_context=y_ctx)
                ensemble_results.append(_predict_single())
            
            self.net.set_context(orig_ctx_x, orig_ctx_y)

            if self.task_type == "cr":
                avg_cifs = []
                for k in range(self.num_events):
                    vals = [res[k].values for res in ensemble_results]
                    avg_cifs.append(pd.DataFrame(np.mean(vals, axis=0), 
                                                index=ensemble_results[0][k].index, 
                                                columns=ensemble_results[0][k].columns))
                return avg_cifs
            else:
                avg_vals = np.mean([res.values for res in ensemble_results], axis=0)
                return pd.DataFrame(avg_vals, index=ensemble_results[0].index, columns=ensemble_results[0].columns)
        else:
            return _predict_single()


