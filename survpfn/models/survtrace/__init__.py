"""survpfn.models.survtrace — SurvTRACE benchmark wrapper.

Wang & Sun, "SurvTRACE: Transformers for Survival Analysis with Competing Events",
ACM BCB 2022 (arXiv:2110.00855).

Public API
----------
    from survpfn.models.survtrace import train_survtrace

    model, risk, surv_probs, surv_times = train_survtrace(
        df_train, df_test, duration_col, event_col
    )
"""
from .survival import train_survtrace

__all__ = ["train_survtrace"]
