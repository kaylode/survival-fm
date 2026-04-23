#!/usr/bin/env bash
# _lib.sh — shared helpers sourced by run_sr.sh and run_cr.sh
# Not meant to be executed directly.

timestamp() { date +"%Y%m%d_%H%M%S"; }

# _extra_args <model-string>
# Emits --epochs / --lr / --device when the model set contains FM or deep models.
_extra_args() {
    local models="$1"
    local args=()
    if echo "$models" | grep -qE "tabpfn|tabdpt|tabicl|deepsurv|mtlr|pchazard|deephit"; then
        args+=("--epochs" "$EPOCHS" "--lr" "$LR" "--device" "$DEVICE")
    fi
    echo "${args[@]}"
}

# run_models <label> <model-string> <dataset-string> [extra_args…]
run_models() {
    local label="$1" models="$2" datasets="$3"
    shift 3
    local extra=("$@")

    mkdir -p "$LOG_DIR"
    local logfile="$LOG_DIR/${label}_$(timestamp).log"

    printf '\n┌─────────────────────────────────────────────────────────────┐\n'
    printf '│  %-61s│\n' "Group   : $label"
    printf '│  %-61s│\n' "Models  : $models"
    printf '│  %-61s│\n' "Data    : $datasets"
    printf '│  %-61s│\n' "Log     : $logfile"
    printf '└─────────────────────────────────────────────────────────────┘\n'

    # shellcheck disable=SC2086
    $PYTHON $SCRIPT \
        --datasets $datasets \
        --models   $models \
        --folds    "$FOLDS" \
        --tune --trials "$TRIALS" \
        --seed     "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --n-ensemble "$N_ENSEMBLE" \
        --save-predictions \
        "${extra[@]}" \
        2>&1 | tee "$logfile"
}

# run_dataset_parallel <label> <model-string> <single-dataset> [extra_args…]
run_dataset_parallel() {
    local label="$1" models="$2" dataset="$3"
    shift 3
    local extra=("$@")

    mkdir -p "$LOG_DIR"
    local logfile="$LOG_DIR/${label}_${dataset}_$(timestamp).log"

    echo "  → Spawning: $dataset ($label) — log: $logfile"

    # shellcheck disable=SC2086
    $PYTHON $SCRIPT \
        --datasets "$dataset" \
        --models   $models \
        --folds    "$FOLDS" \
        --tune --trials "$TRIALS" \
        --seed     "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        "${extra[@]}" \
        >"$logfile" 2>&1 &
}
