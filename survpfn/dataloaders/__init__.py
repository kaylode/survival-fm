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
    load_sirbu_cv,
    load_sirbu_mi,
    load_sirbu_stroke,
    load_sirbu_mortality
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
    "SIRBU_CV": load_sirbu_cv,
    "SIRBU_MI": load_sirbu_mi,
    "SIRBU_STROKE": load_sirbu_stroke,
    "SIRBU_MORTALITY": load_sirbu_mortality,
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
