#!/usr/bin/env bash
# DLP-Affinity 训练→多突变组合→预测 投递
# 模板注册：./argo/dlp-affinity-template.create.sh train_predict
#
# 用法：
#   ./argo/dlp-affinity-train_predict.submit.sh
#   NUM_EPOCHS=20 EXP_NAME=tp001 ./argo/dlp-affinity-train_predict.submit.sh
#   LABEL_COL=kd BETTER_DIRECTION=lower ./argo/dlp-affinity-train_predict.submit.sh
set -euo pipefail

TEMPLATE_NAME="dlp-affinity-train-predict"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train-predict_$(date +%Y%m%d_%H%M%S)}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
DEFAULT_TRAIN="${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_train_model_input.csv"
TRAIN_PATH_RESOLVED="${TRAIN_PATH:-${DEFAULT_TRAIN}}"
SINGLE_MUTANT_RESOLVED="${SINGLE_MUTANT_PATH:-${TRAIN_PATH_RESOLVED}}"
BETTER_DIRECTION_RESOLVED="${BETTER_DIRECTION:-lower}"

echo "=== DLP-Affinity train_predict submit ==="
echo "template:          ${TEMPLATE_NAME}"
echo "namespace:         ${NAMESPACE}"
echo "exp-name:          ${EXP_NAME}"
echo "dag:               build-combinations(validate) -> train -> predict"
echo "train-path:        ${TRAIN_PATH_RESOLVED}"
echo "single-mutant-path:${SINGLE_MUTANT_RESOLVED}"
echo "better-direction:  ${BETTER_DIRECTION_RESOLVED}"
echo "combo/predict out: ${OUTPUT_DIR}/${EXP_NAME}/combo/"

# 不用关联数组，兼容 macOS Bash 3.2
ARGS=(--from "workflowtemplate/${TEMPLATE_NAME}" -n "${NAMESPACE}")
ARGS+=(-p "train-path=${TRAIN_PATH_RESOLVED}")
ARGS+=(-p "val-path=${VAL_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_val_model_input.csv}")
ARGS+=(-p "esm-model-path=${ESM_MODEL_PATH:-/mnt/nas1/liubo/models/esm2_t30_150M_UR50D}")
ARGS+=(-p "output-dir=${OUTPUT_DIR}")
ARGS+=(-p "work-root=${WORK_ROOT}")
ARGS+=(-p "exp-name=${EXP_NAME}")
ARGS+=(-p "num-epochs=${NUM_EPOCHS}")
ARGS+=(-p "seed=${SEED:-42}")
ARGS+=(-p "device=${DEVICE:-cuda}")
ARGS+=(-p "freeze-esm=${FREEZE_ESM:-true}")
ARGS+=(-p "use-small=${USE_SMALL:-false}")
ARGS+=(-p "gpu-model-series=${GPU_MODEL_SERIES:-A10}")
ARGS+=(-p "single-mutant-path=${SINGLE_MUTANT_RESOLVED}")
ARGS+=(-p "wt-path=${WT_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_wildtype_ref.csv}")
ARGS+=(-p "label-col=${LABEL_COL:-kd}")
ARGS+=(-p "better-direction=${BETTER_DIRECTION_RESOLVED}")
ARGS+=(-p "min-order=${MIN_ORDER:-2}")
ARGS+=(-p "max-order=${MAX_ORDER:-0}")
ARGS+=(-p "max-combinations=${MAX_COMBINATIONS:-0}")
ARGS+=(-p "batch-size=${BATCH_SIZE:-8}")
[ -n "${IMAGE:-}" ] && ARGS+=(-p "image=${IMAGE}")
[ -n "${NAS_MOUNT_PATH:-}" ] && ARGS+=(-p "nas-mount-path=${NAS_MOUNT_PATH}")
[ -n "${ESM_CHECKPOINT:-}" ] && ARGS+=(-p "esm-checkpoint=${ESM_CHECKPOINT}")
[ -n "${CONFIG_PATH:-}" ] && ARGS+=(-p "config-path=${CONFIG_PATH}")
[ -n "${COMBO_OUTPUT_PATH:-}" ] && ARGS+=(-p "combo-output-path=${COMBO_OUTPUT_PATH}")
[ -n "${PREDICT_OUTPUT_PATH:-}" ] && ARGS+=(-p "predict-output-path=${PREDICT_OUTPUT_PATH}")

argo submit "${ARGS[@]}" "$@"
echo "submit done."
echo "  status: argo get -n ${NAMESPACE} @latest"
echo "  logs:   argo logs -n ${NAMESPACE} @latest"
echo "  hint:   确保 ${WORK_ROOT}/script/build_multimutant_library.py 已同步到 NAS"
