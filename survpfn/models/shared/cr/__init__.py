"""survpfn.models.shared.cr — Competing-risks training utilities.

Public API
----------
CRDeepHitLoss         — Fixed DeepHit CR loss (strict censoring mask P(T > C)).
CRCoxLoss             — Vectorised cause-specific Cox partial-likelihood.
BaseCRJointFinetune   — BaseJointSurvFinetune subclass; overrides 3 template methods.
TabPFNCRPH            — TabPFN backbone + CR head (DeepHit or Cox).
TabDPTCRPH            — TabDPT backbone + CR head.
TabICLCRPH            — TabICL backbone + CR head.
CRZeroShotPredictor   — Zero-shot ICL for competing risks; correct pycox orientation.
"""

from .loss import CRDeepHitLoss, CRCoxLoss
from .finetune import BaseCRJointFinetune, TabPFNCRPH, TabDPTCRPH, TabICLCRPH
from .zeroshot import CRZeroShotPredictor

__all__ = [
    "CRDeepHitLoss",
    "CRCoxLoss",
    "BaseCRJointFinetune",
    "TabPFNCRPH",
    "TabDPTCRPH",
    "TabICLCRPH",
    "CRZeroShotPredictor",
]
