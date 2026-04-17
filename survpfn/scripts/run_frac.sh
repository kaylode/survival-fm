#!/usr/bin/env bash

FOLDS=5
TRIALS=20
SEED=42
OUTPUT_DIR="results/benchmark_frac"
LOG_DIR="logs"

EPOCHS=50
LR="1e-4"
DEVICE="cuda:0"
BATCH_SIZE=128
N_ENSEMBLE=5

uv run survpfn/scripts/benchmark.py \
    --datasets EICU_SURV MIMIC_SURV_B \
    --models cox rsf gbsa deepsurv dysurv mtlr pchazard deephit_single tabpfn_zeroshot_perbin_time_ens tabpfn_embedding_deephit tabpfn_finetune  \
    --label-fractions 0.01 0.02 0.05 0.1 \
	--n-ensemble "$N_ENSEMBLE" \
	--folds    "$FOLDS" \
	--seed     "$SEED" \
	--output-dir "$OUTPUT_DIR" \
	--batch-size "$BATCH_SIZE"
	# --tune --trials "$TRIALS" \

