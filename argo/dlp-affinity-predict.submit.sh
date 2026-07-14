#!/usr/bin/env bash
# DLP-Affinity 预测投递：向已注册的 WorkflowTemplate 提交 Workflow
# 模板注册：./argo/dlp-affinity-template.create.sh predict
#
# 用法：
#   ./argo/dlp-affinity-predict.submit.sh
#   EXP_NAME=dlp-affinity-train_20260707_105354 ./argo/dlp-affinity-predict.submit.sh
#   CHECKPOINT=/mnt/.../best_model.pt ./argo/dlp-affinity-predict.submit.sh
set -euo pipefail

TEMPLATE_NAME="dlp-affinity-predict"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${EXP_NAME}/best_model.pt}"
INPUT_PATH="${INPUT_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_val_model_input.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/predictions_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).csv}"

echo "=== DLP-Affinity predict submit ==="
echo "template:    ${TEMPLATE_NAME}"
echo "namespace:   ${NAMESPACE}"
echo "checkpoint:  ${CHECKPOINT}"
echo "input-path:  ${INPUT_PATH}"
echo "output-path: ${OUTPUT_PATH}"

# 不用关联数组，兼容 macOS Bash 3.2
ARGS=(--from "workflowtemplate/${TEMPLATE_NAME}" -n "${NAMESPACE}")
ARGS+=(-p "checkpoint=${CHECKPOINT}")
ARGS+=(-p "input-path=${INPUT_PATH}")
ARGS+=(-p "output-path=${OUTPUT_PATH}")
ARGS+=(-p "work-root=${WORK_ROOT}")
ARGS+=(-p "batch-size=${BATCH_SIZE:-8}")
ARGS+=(-p "device=${DEVICE:-cuda}")
ARGS+=(-p "gpu-model-series=${GPU_MODEL_SERIES:-A10}")
[ -n "${IMAGE:-}" ] && ARGS+=(-p "image=${IMAGE}")
[ -n "${NAS_MOUNT_PATH:-}" ] && ARGS+=(-p "nas-mount-path=${NAS_MOUNT_PATH}")

argo submit "${ARGS[@]}" "$@"
echo "submit done."
echo "  status: argo get -n ${NAMESPACE} @latest"
echo "  logs:   argo logs -n ${NAMESPACE} @latest | tail -50"
