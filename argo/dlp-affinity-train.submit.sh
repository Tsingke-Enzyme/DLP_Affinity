#!/usr/bin/env bash
# DLP-Affinity 训练微调投递脚本
# 输入：环境变量覆盖模版参数（可选）；输出：向 Argo 提交 Workflow 并打印工作流名
# 处理逻辑：apply WorkflowTemplate → argo submit --from workflowtemplate/dlp-affinity-train
#
# 用法：
#   ./argo/dlp-affinity-train.submit.sh
#   EXP_NAME=exp001 FREEZE_ESM=true ./argo/dlp-affinity-train.submit.sh
#   ./argo/dlp-affinity-train.submit.sh -p use-small=true
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/dlp-affinity-train.yaml"
TEMPLATE_NAME="dlp-affinity-train"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
NAS_MOUNT_PATH="${NAS_MOUNT_PATH:-/mnt}"
IMAGE="${IMAGE:-beijing-acr-cr-registry-vpc.cn-beijing.cr.aliyuncs.com/biocloud/enzyme_dev:DLP_Affinity.v1.0.0}"
TRAIN_PATH="${TRAIN_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_train_model_input.csv}"
VAL_PATH="${VAL_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_val_model_input.csv}"
ESM_MODEL_PATH="${ESM_MODEL_PATH:-/mnt/nas1/liubo/models/esm2_t30_150M_UR50D}"
ESM_CHECKPOINT="${ESM_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
FREEZE_ESM="${FREEZE_ESM:-true}"
USE_SMALL="${USE_SMALL:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"
GPU_MODEL_SERIES="${GPU_MODEL_SERIES:-A10}"

echo "=== DLP-Affinity train submit ==="
echo "namespace:        ${NAMESPACE}"
echo "template:         ${TEMPLATE_NAME}"
echo "image:            ${IMAGE}"
echo "train-path:       ${TRAIN_PATH}"
echo "val-path:         ${VAL_PATH}"
echo "esm-model-path:   ${ESM_MODEL_PATH}"
echo "esm-checkpoint:   ${ESM_CHECKPOINT:-<empty>}"
echo "output-dir:       ${OUTPUT_DIR}"
echo "work-root:        ${WORK_ROOT}"
echo "exp-name:         ${EXP_NAME}"
echo "best-model:       ${OUTPUT_DIR}/${EXP_NAME}/best_model.pt"
echo "freeze-esm:       ${FREEZE_ESM}"
echo "use-small:        ${USE_SMALL}"
echo "gpu-model-series: ${GPU_MODEL_SERIES}"

kubectl apply -n "${NAMESPACE}" -f "${TEMPLATE_FILE}" >/dev/null

ARGS=(
  --from "workflowtemplate/${TEMPLATE_NAME}"
  -n "${NAMESPACE}"
  -p "nas-mount-path=${NAS_MOUNT_PATH}"
  -p "image=${IMAGE}"
  -p "train-path=${TRAIN_PATH}"
  -p "val-path=${VAL_PATH}"
  -p "esm-model-path=${ESM_MODEL_PATH}"
  -p "esm-checkpoint=${ESM_CHECKPOINT}"
  -p "output-dir=${OUTPUT_DIR}"
  -p "work-root=${WORK_ROOT}"
  -p "exp-name=${EXP_NAME}"
  -p "seed=${SEED}"
  -p "device=${DEVICE}"
  -p "freeze-esm=${FREEZE_ESM}"
  -p "use-small=${USE_SMALL}"
  -p "config-path=${CONFIG_PATH}"
  -p "gpu-model-series=${GPU_MODEL_SERIES}"
)

argo submit "${ARGS[@]}" "$@"
echo "submit done. watch: argo watch -n ${NAMESPACE} @latest"
echo "logs:  argo logs -n ${NAMESPACE} @latest -f"
