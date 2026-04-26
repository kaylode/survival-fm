#!/usr/bin/env bash
set -euo pipefail

# Define datasets and models to iterate over
DATASETS="MIMIC_SURV_B EICU_SURV SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN SEER"

FM_EMBEDDING="tabicl_embedding_mtlr_hybrid tabdpt_embedding_mtlr_hybrid tabpfn_embedding_mtlr_hybrid tabicl_embedding_deephit tabdpt_embedding_deephit tabpfn_embedding_deephit tabicl_embedding_cox tabdpt_embedding_cox tabpfn_embedding_cox"
FINETUNE="tabpfn_finetune_hybrid tabdpt_finetune_hybrid tabicl_finetune_hybrid"
ZEROSHOT_TEMPORAL="tabpfn_zeroshot_hybrid tabdpt_zeroshot_hybrid tabicl_zeroshot_hybrid"

MODELS="$FM_EMBEDDING $FINETUNE $ZEROSHOT_TEMPORAL"

echo "Generating calibration plots..."

for DATASET in $DATASETS; do
    mkdir -p "results/xai/calibration/${DATASET}"
    
    for MODEL in $MODELS; do
        # Check if predictions exist before trying to plot
        if [ -d "results/predictions/${DATASET}/${MODEL}" ]; then
            echo "Plotting: $DATASET / $MODEL"
            uv run survpfn/scripts/plot_calibration.py \
                --dataset "$DATASET" \
                --model   "$MODEL" \
                --out     "results/xai/calibration/${DATASET}/${MODEL}_calibr.pdf"
        else
            echo "Skipping: $DATASET / $MODEL (No predictions found)"
        fi
    done
done

echo "Done."