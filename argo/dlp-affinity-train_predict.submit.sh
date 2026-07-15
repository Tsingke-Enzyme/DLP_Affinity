#!/usr/bin/env bash
# DLP-Affinity 训练→多突变组合→预测 投递
# 模板注册：./argo/dlp-affinity-template.create.sh train_predict
#
# 文件入口：TRAIN_PATH / SINGLE_MUTANT_PATH / WT_PATH / OUTPUT_DIR；可选 VAL_PATH（默认=TRAIN_PATH）
# 派生：OUTPUT_DIR/{train,combo,predict}/
#
# 用法：
#   ./argo/dlp-affinity-train_predict.submit.sh
#   OUTPUT_DIR=/mnt/.../run001 NUM_EPOCHS=20 ./argo/dlp-affinity-train_predict.submit.sh
#   WT_PATH=/path/to/wt.csv TOP_N_SINGLES=30 ./argo/dlp-affinity-train_predict.submit.sh
#   VAL_PATH=/path/to/val.csv ./argo/dlp-affinity-train_predict.submit.sh
set -euo pipefail

TEMPLATE_NAME="dlp-affinity-train-predict"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs/dlp-affinity-train-predict_$(date +%Y%m%d_%H%M%S)}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
DEFAULT_TRAIN="${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_train_model_input.csv"
TRAIN_PATH_RESOLVED="${TRAIN_PATH:-${DEFAULT_TRAIN}}"
# 未单独提供 VAL_PATH 时与 train-path 相同，保证训练写出 best_model.pt
VAL_PATH_RESOLVED="${VAL_PATH:-${TRAIN_PATH_RESOLVED}}"
SINGLE_MUTANT_RESOLVED="${SINGLE_MUTANT_PATH:-${TRAIN_PATH_RESOLVED}}"
BETTER_DIRECTION_RESOLVED="${BETTER_DIRECTION:-lower}"
TOP_N_SINGLES_RESOLVED="${TOP_N_SINGLES:-10}"

echo "=== DLP-Affinity train_predict submit ==="
echo "template:          ${TEMPLATE_NAME}"
echo "namespace:         ${NAMESPACE}"
echo "dag:               build-combinations -> train -> predict"
echo "train-path:        ${TRAIN_PATH_RESOLVED}"
echo "val-path:          ${VAL_PATH_RESOLVED}"
echo "single-mutant-path:${SINGLE_MUTANT_RESOLVED}"
echo "wt-path:           ${WT_PATH:-<empty, use consensus+top-n>}"
echo "output-dir:        ${OUTPUT_DIR}"
echo "  train ->         ${OUTPUT_DIR}/train/"
echo "  combo ->         ${OUTPUT_DIR}/combo/"
echo "  predict ->       ${OUTPUT_DIR}/predict/"
echo "top-n-singles:     ${TOP_N_SINGLES_RESOLVED}"

ARGS=(--from "workflowtemplate/${TEMPLATE_NAME}" -n "${NAMESPACE}")
ARGS+=(-p "train-path=${TRAIN_PATH_RESOLVED}")
ARGS+=(-p "val-path=${VAL_PATH_RESOLVED}")
ARGS+=(-p "single-mutant-path=${SINGLE_MUTANT_RESOLVED}")
ARGS+=(-p "output-dir=${OUTPUT_DIR}")
ARGS+=(-p "num-epochs=${NUM_EPOCHS}")
ARGS+=(-p "seed=${SEED:-42}")
ARGS+=(-p "device=${DEVICE:-cuda}")
ARGS+=(-p "freeze-esm=${FREEZE_ESM:-true}")
ARGS+=(-p "use-small=${USE_SMALL:-false}")
ARGS+=(-p "gpu-model-series=${GPU_MODEL_SERIES:-A10}")
ARGS+=(-p "label-col=${LABEL_COL:-kd}")
ARGS+=(-p "better-direction=${BETTER_DIRECTION_RESOLVED}")
ARGS+=(-p "top-n-singles=${TOP_N_SINGLES_RESOLVED}")
ARGS+=(-p "min-order=${MIN_ORDER:-2}")
ARGS+=(-p "max-order=${MAX_ORDER:-5}")
ARGS+=(-p "max-combinations=${MAX_COMBINATIONS:-50000}")
ARGS+=(-p "batch-size=${BATCH_SIZE:-8}")
[ -n "${WT_PATH:-}" ] && ARGS+=(-p "wt-path=${WT_PATH}")
[ -n "${IMAGE:-}" ] && ARGS+=(-p "image=${IMAGE}")
[ -n "${NAS_MOUNT_PATH:-}" ] && ARGS+=(-p "nas-mount-path=${NAS_MOUNT_PATH}")

argo submit "${ARGS[@]}" "$@"
echo "submit done."
echo "  status: argo get -n ${NAMESPACE} @latest"
echo "  logs:   argo logs -n ${NAMESPACE} @latest"
echo "  hint:   确保 ${WORK_ROOT}/script/build_multimutant_library.py 已同步到 NAS"
