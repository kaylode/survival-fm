"""survpfn.data — data loading, preprocessing, and benchmark dataset loaders."""
from typing import *

from .single import (
	load_support,
	load_metabric,
	load_gbsg,
	load_whas500,
	load_veterans,
	load_flchain,
	load_seer_dataset,
	load_urrah_dataset,
	load_survset_dataset, SURVSET_BENCHMARK,
    load_ormoni_tirodei_cv,
    load_ormoni_tirodei_mi,
    load_ormoni_tirodei_stroke,
    load_ormoni_tirodei_mortality,
    temporal_split,
    load_eicu_survival,
    load_mimiciv_survival_B,
)

from .competing import (
    load_framingham,
    load_pbc2,
    load_support_cr,
    load_synthetic_cr,
)



BENCHMARK_DATASETS = {
    # Public — pycox
    "SUPPORT2":     load_support,
    "METABRIC":     load_metabric,
    "GBSG":         load_gbsg,
    # Public — scikit-survival
    "WHAS500":      load_whas500,
    "VETERANS":     load_veterans,
    "FLCHAIN":      load_flchain,
    # SEER Breast Cancer (public CSV, ~4024 patients)
    "SEER":         load_seer_dataset,
    # SurvSet — 25-dataset curated benchmark (SS_ prefix)
    "ORMONI_TIRODEI_CV": load_ormoni_tirodei_cv,
    "ORMONI_TIRODEI_MI": load_ormoni_tirodei_mi,
    "ORMONI_TIRODEI_STROKE": load_ormoni_tirodei_stroke,
    "ORMONI_TIRODEI_MORTALITY": load_ormoni_tirodei_mortality,
    "URRAH": load_urrah_dataset,
    # Large-scale ICU survival datasets
    "EICU_SURV":     load_eicu_survival,
    "MIMIC_SURV_B":  load_mimiciv_survival_B,
    **{"SS_" + k.upper(): (lambda k=k: load_survset_dataset(k)) for k in SURVSET_BENCHMARK}
}

CR_DATASETS = {
    "FRAMINGHAM": load_framingham,
    "PBC2": load_pbc2,
    "SUPPORT_CR": load_support_cr,
    "SYNTHETIC_CR": load_synthetic_cr,
}

ALL_DATASETS = {**BENCHMARK_DATASETS, **CR_DATASETS}

def get_dataset(name):
    if name in ALL_DATASETS:
        return ALL_DATASETS[name]()
    else:
        raise ValueError(f"Dataset {name} not found.")

__all__ = [
    "get_dataset",
    "BENCHMARK_DATASETS",
    "CR_DATASETS",
    "ALL_DATASETS"
]
