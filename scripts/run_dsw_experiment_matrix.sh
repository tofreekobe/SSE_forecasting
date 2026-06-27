#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/workspace/sse}"
PACKAGE_DIR="${PACKAGE_DIR:-/mnt/workspace/hf_dataset_package}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/sse_outputs/experiment_matrix_$(date +%Y%m%d_%H%M%S)}"
MATRIX_PROFILE="${MATRIX_PROFILE:-full}"
PROTOCOLS="${PROTOCOLS:-random blocked}"

EPOCHS_MAIN="${EPOCHS_MAIN:-50}"
EPOCHS_COMPARE="${EPOCHS_COMPARE:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0007}"
TRAIN_EVAL_MAX_BATCHES="${TRAIN_EVAL_MAX_BATCHES:-32}"

MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-0}"
MAX_VAL_EVENTS="${MAX_VAL_EVENTS:-0}"
MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-0}"
FORCE="${FORCE:-0}"

export PYTHONUTF8=1
mkdir -p "${OUTPUT_ROOT}/logs"

if [[ ! -f "${PACKAGE_DIR}/manifest.csv" ]]; then
  echo "Missing dataset package manifest: ${PACKAGE_DIR}/manifest.csv" >&2
  exit 2
fi

cd "${PROJECT_DIR}"

python -m pytest tests/test_forecast_contract.py -q
python -m py_compile \
  scripts/train_forecast_model.py \
  scripts/collect_experiment_matrix.py \
  scripts/plot_forecast_examples.py \
  scripts/summarize_training_results.py

python scripts/run_baseline_forecasts.py \
  --package-dir "${PACKAGE_DIR}" \
  --output-dir "${OUTPUT_ROOT}/baselines" \
  --split both \
  --forecast-start 60 \
  --horizons 1 5 10 30 50 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/baselines.log"

LIMIT_ARGS=()
if [[ "${MAX_TRAIN_EVENTS}" != "0" ]]; then
  LIMIT_ARGS+=(--max-train-events "${MAX_TRAIN_EVENTS}")
fi
if [[ "${MAX_VAL_EVENTS}" != "0" ]]; then
  LIMIT_ARGS+=(--max-val-events "${MAX_VAL_EVENTS}")
fi
if [[ "${MAX_TEST_EVENTS}" != "0" ]]; then
  LIMIT_ARGS+=(--max-test-events "${MAX_TEST_EVENTS}")
fi

run_train() {
  local run_name="$1"
  local protocol="$2"
  local model_type="$3"
  local input_mode="$4"
  local m0_loss_weight="$5"
  local epochs="$6"
  local run_root="${OUTPUT_ROOT}/${run_name}"
  local metrics_path="${run_root}/${protocol}/metrics.json"
  local log_path="${OUTPUT_ROOT}/logs/${run_name}.${protocol}.log"

  if [[ -f "${metrics_path}" && "${FORCE}" != "1" ]]; then
    echo "Skipping existing run: ${run_name}/${protocol}"
    return
  fi

  echo "=== $(date -Iseconds) ${run_name}/${protocol} model=${model_type} input=${input_mode} m0=${m0_loss_weight} epochs=${epochs} ===" \
    | tee "${log_path}"

  python scripts/train_forecast_model.py \
    --package-dir "${PACKAGE_DIR}" \
    --output-dir "${run_root}" \
    --protocol "${protocol}" \
    --forecast-start 60 \
    --forecast-horizon 50 \
    "${LIMIT_ARGS[@]}" \
    --epochs "${epochs}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --model-type "${model_type}" \
    --input-mode "${input_mode}" \
    --device cuda \
    --lr "${LEARNING_RATE}" \
    --active-weight 1.0 \
    --m0-loss-weight "${m0_loss_weight}" \
    --train-eval-max-batches "${TRAIN_EVAL_MAX_BATCHES}" \
    --amp \
    --tensorboard-dir off \
    --log-every 1 \
    2>&1 | tee -a "${log_path}"
}

declare -a RUNS_FULL=(
  "main_residual_full segmented_residual full 0.005 ${EPOCHS_MAIN}"
  "model_segmented_full segmented full 0.005 ${EPOCHS_COMPARE}"
  "model_plain_full plain full 0.005 ${EPOCHS_COMPARE}"
  "ablate_no_gnss segmented_residual no_gnss 0.005 ${EPOCHS_COMPARE}"
  "ablate_gnss_only segmented_residual gnss_only 0.005 ${EPOCHS_COMPARE}"
  "ablate_last_slip_only segmented_residual last_slip_only 0.005 ${EPOCHS_COMPARE}"
  "ablate_no_m0_loss segmented_residual full 0.0 ${EPOCHS_COMPARE}"
)

declare -a RUNS_CORE=(
  "main_residual_full segmented_residual full 0.005 ${EPOCHS_MAIN}"
  "ablate_no_gnss segmented_residual no_gnss 0.005 ${EPOCHS_COMPARE}"
  "ablate_gnss_only segmented_residual gnss_only 0.005 ${EPOCHS_COMPARE}"
)

if [[ "${MATRIX_PROFILE}" == "core" ]]; then
  RUNS=("${RUNS_CORE[@]}")
else
  RUNS=("${RUNS_FULL[@]}")
fi

for run_spec in "${RUNS[@]}"; do
  read -r run_name model_type input_mode m0_loss_weight epochs <<< "${run_spec}"
  for protocol in ${PROTOCOLS}; do
    run_train "${run_name}" "${protocol}" "${model_type}" "${input_mode}" "${m0_loss_weight}" "${epochs}"
    python scripts/collect_experiment_matrix.py --matrix-dir "${OUTPUT_ROOT}" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/collect.latest.log"
  done
done

python scripts/collect_experiment_matrix.py --matrix-dir "${OUTPUT_ROOT}"
echo "Experiment matrix complete: ${OUTPUT_ROOT}"
