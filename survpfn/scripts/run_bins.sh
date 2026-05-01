#!/usr/bin/env bash

FOLDS=3
TRIALS=20
SEED=42
OUTPUT_DIR="results/benchmark_bins"
LOG_DIR="logs"

EPOCHS=50
LR="1e-4"
DEVICE="cuda:0"
BATCH_SIZE=128
N_ENSEMBLE=5

uv run survpfn/scripts/benchmark.py \
    --datasets WHAS500 METABRIC EICU_SURV \
    --models mtlr deephit_single tabpfn_zeroshot_perbin_time_ens tabpfn_embedding_deephit tabpfn_finetune tabdpt_zeroshot_perbin_time_ens tabdpt_embedding_deephit tabdpt_finetune tabicl_zeroshot_perbin_time_ens tabicl_embedding_deephit tabicl_finetune \
	--n-ensemble "$N_ENSEMBLE" \
	--folds    "$FOLDS" \
	--seed     "$SEED" \
	--output-dir "$OUTPUT_DIR" \
	--batch-size "$BATCH_SIZE" \
	--tuned-dir "results/benchmark" \
	--num-durations $1
	# 5 10 30 50 70 100
	# --tune --trials "$TRIALS" \

