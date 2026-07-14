#!/usr/bin/env bash
# DLP-Affinity 训练→多突变组合→预测 投递
# 模板注册：./argo/dlp-affinity-template.create.sh train_predict
#
# 用法：
#   ./argo/dlp-affinity-train_predict.submit.sh
#   NUM_EPOCHS=20 EXP_NAME=tp001 ./argo/dlp-affinity-train_predict.submit.sh
#   LABEL_COL=kd BETTER_DIRECTION=lower ./argo/dlp-affinity-train_predict.submit.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_NAME="dlp-affinity-train-predict"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train-predict_$(date +%Y%m%d_%H%M%S)}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
DEFAULT_TRAIN="${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_train_model_input.csv"

declare -A OVERRIDES=(
  [train-path]="${TRAIN_PATH:-${DEFAULT_TRAIN}}"
  [val-path]="${VAL_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_val_model_input.csv}"
  [esm-model-path]="${ESM_MODEL_PATH:-/mnt/nas1/liubo/models/esm2_t30_150M_UR50D}"
  [output-dir]="${OUTPUT_DIR}"
  [work-root]="${WORK_ROOT}"
  [exp-name]="${EXP_NAME}"
  [num-epochs]="${NUM_EPOCHS}"
  [seed]="${SEED:-42}"
  [device]="${DEVICE:-cuda}"
  [freeze-esm]="${FREEZE_ESM:-true}"
  [use-small]="${USE_SMALL:-false}"
  [gpu-model-series]="${GPU_MODEL_SERIES:-A10}"
  [single-mutant-path]="${SINGLE_MUTANT_PATH:-${TRAIN_PATH:-${DEFAULT_TRAIN}}}"
  [wt-path]="${WT_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_wildtype_ref.csv}"
  [label-col]="${LABEL_COL:-kd}"
  [better-direction]="${BETTER_DIRECTION:-lower}"
  [min-order]="${MIN_ORDER:-2}"
  [max-order]="${MAX_ORDER:-0}"
  [max-combinations]="${MAX_COMBINATIONS:-0}"
  [batch-size]="${BATCH_SIZE:-8}"
)
[ -n "${IMAGE:-}" ] && OVERRIDES[image]="${IMAGE}"
[ -n "${NAS_MOUNT_PATH:-}" ] && OVERRIDES[nas-mount-path]="${NAS_MOUNT_PATH}"
[ -n "${ESM_CHECKPOINT:-}" ] && OVERRIDES[esm-checkpoint]="${ESM_CHECKPOINT}"
[ -n "${CONFIG_PATH:-}" ] && OVERRIDES[config-path]="${CONFIG_PATH}"
[ -n "${COMBO_OUTPUT_PATH:-}" ] && OVERRIDES[combo-output-path]="${COMBO_OUTPUT_PATH}"
[ -n "${PREDICT_OUTPUT_PATH:-}" ] && OVERRIDES[predict-output-path]="${PREDICT_OUTPUT_PATH}"

echo "=== DLP-Affinity train_predict submit ==="
echo "template:          ${TEMPLATE_NAME}"
echo "namespace:         ${NAMESPACE}"
echo "exp-name:          ${EXP_NAME}"
echo "train-path:        ${OVERRIDES[train-path]}"
echo "single-mutant-path:${OVERRIDES[single-mutant-path]}"
echo "better-direction:  ${OVERRIDES[better-direction]}"
echo "combo/predict out: ${OUTPUT_DIR}/${EXP_NAME}/combo/"

ARGS=(--from "workflowtemplate/${TEMPLATE_NAME}" -n "${NAMESPACE}")
for key in "${!OVERRIDES[@]}"; do
  ARGS+=(-p "${key}=${OVERRIDES[$key]}")
done

argo submit "${ARGS[@]}" "$@"
echo "submit done."
echo "  status: argo get -n ${NAMESPACE} @latest"
echo "  logs:   argo logs -n ${NAMESPACE} @latest"
echo "  hint:   确保 ${WORK_ROOT}/script/build_multimutant_library.py 已同步到 NAS"
