#!/usr/bin/env bash
# DLP-Affinity 预测投递脚本
# 输入：环境变量覆盖模版参数（可选）；输出：向 Argo 提交 Workflow 并打印工作流名
# 处理逻辑：apply WorkflowTemplate → argo submit --from workflowtemplate/dlp-affinity-predict
#
# 用法：
#   ./argo/dlp-affinity-predict.submit.sh
#   CHECKPOINT=/mnt/.../best_model.pt INPUT_PATH=/mnt/.../test.csv ./argo/dlp-affinity-predict.submit.sh
#   EXP_NAME=dlp-affinity-train_20260706_120000 ./argo/dlp-affinity-predict.submit.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/dlp-affinity-predict.yaml"
TEMPLATE_NAME="dlp-affinity-predict"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
NAS_MOUNT_PATH="${NAS_MOUNT_PATH:-/mnt}"
IMAGE="${IMAGE:-beijing-acr-cr-registry-vpc.cn-beijing.cr.aliyuncs.com/biocloud/enzyme_dev:DLP_Affinity.v1.0.0}"
# 若设置 EXP_NAME，默认使用 ${OUTPUT_DIR}/${EXP_NAME}/best_model.pt
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${EXP_NAME}/best_model.pt}"
INPUT_PATH="${INPUT_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_val_model_input.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/predictions_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
GPU_MODEL_SERIES="${GPU_MODEL_SERIES:-A10}"

echo "=== DLP-Affinity predict submit ==="
echo "namespace:        ${NAMESPACE}"
echo "template:         ${TEMPLATE_NAME}"
echo "image:            ${IMAGE}"
echo "checkpoint:       ${CHECKPOINT}"
echo "input-path:       ${INPUT_PATH}"
echo "output-path:      ${OUTPUT_PATH}"
echo "work-root:        ${WORK_ROOT}"
echo "batch-size:       ${BATCH_SIZE}"
echo "gpu-model-series: ${GPU_MODEL_SERIES}"

kubectl apply -n "${NAMESPACE}" -f "${TEMPLATE_FILE}" >/dev/null

ARGS=(
  --from "workflowtemplate/${TEMPLATE_NAME}"
  -n "${NAMESPACE}"
  -p "nas-mount-path=${NAS_MOUNT_PATH}"
  -p "image=${IMAGE}"
  -p "checkpoint=${CHECKPOINT}"
  -p "input-path=${INPUT_PATH}"
  -p "output-path=${OUTPUT_PATH}"
  -p "work-root=${WORK_ROOT}"
  -p "batch-size=${BATCH_SIZE}"
  -p "device=${DEVICE}"
  -p "gpu-model-series=${GPU_MODEL_SERIES}"
)

argo submit "${ARGS[@]}" "$@"
echo "submit done. watch: argo watch -n ${NAMESPACE} @latest"
echo "logs:  argo logs -n ${NAMESPACE} @latest -f"
