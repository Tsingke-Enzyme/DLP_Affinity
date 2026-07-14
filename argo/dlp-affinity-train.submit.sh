#!/usr/bin/env bash
# DLP-Affinity 训练投递：向已注册的 WorkflowTemplate 提交 Workflow
# 模板注册：./argo/dlp-affinity-template.create.sh train
#
# 用法：
#   ./argo/dlp-affinity-train.submit.sh
#   NUM_EPOCHS=20 EXP_NAME=exp001 ./argo/dlp-affinity-train.submit.sh
#   ./argo/dlp-affinity-train.submit.sh -p num-epochs=10
set -euo pipefail

TEMPLATE_NAME="dlp-affinity-train"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train_$(date +%Y%m%d_%H%M%S)}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"

echo "=== DLP-Affinity train submit ==="
echo "template:   ${TEMPLATE_NAME}"
echo "namespace:  ${NAMESPACE}"
echo "exp-name:   ${EXP_NAME}"
echo "num-epochs: ${NUM_EPOCHS}"
echo "best-model: ${OUTPUT_DIR}/${EXP_NAME}/best_model.pt"

# 不用关联数组，兼容 macOS Bash 3.2
ARGS=(--from "workflowtemplate/${TEMPLATE_NAME}" -n "${NAMESPACE}")
ARGS+=(-p "train-path=${TRAIN_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_train_model_input.csv}")
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
[ -n "${IMAGE:-}" ] && ARGS+=(-p "image=${IMAGE}")
[ -n "${NAS_MOUNT_PATH:-}" ] && ARGS+=(-p "nas-mount-path=${NAS_MOUNT_PATH}")
[ -n "${ESM_CHECKPOINT:-}" ] && ARGS+=(-p "esm-checkpoint=${ESM_CHECKPOINT}")
[ -n "${CONFIG_PATH:-}" ] && ARGS+=(-p "config-path=${CONFIG_PATH}")

argo submit "${ARGS[@]}" "$@"
echo "submit done."
echo "  status: argo get -n ${NAMESPACE} @latest"
echo "  logs:   argo logs -n ${NAMESPACE} @latest | rg 'INFO -|ERROR|Training completed'"
