#!/usr/bin/env bash
# DLP-Affinity 训练投递：向已注册的 WorkflowTemplate 提交 Workflow
# 模板注册：./argo/dlp-affinity-template.create.sh train
#
# 用法：
#   ./argo/dlp-affinity-train.submit.sh
#   NUM_EPOCHS=20 EXP_NAME=exp001 ./argo/dlp-affinity-train.submit.sh
#   ./argo/dlp-affinity-train.submit.sh -p num-epochs=10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_NAME="dlp-affinity-train"
NAMESPACE="${NAMESPACE:-default}"

WORK_ROOT="${WORK_ROOT:-/mnt/nas1/liubo/project/DLP_Affinity}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/outputs}"
EXP_NAME="${EXP_NAME:-dlp-affinity-train_$(date +%Y%m%d_%H%M%S)}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"

# 仅覆盖与默认值不同的业务参数；基础设施参数走模板默认值，可用 -p 追加
declare -A OVERRIDES=(
  [train-path]="${TRAIN_PATH:-${WORK_ROOT}/release_package/data/7KMG/LY-CoV555_DMS_train_model_input.csv}"
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
)
[ -n "${IMAGE:-}" ] && OVERRIDES[image]="${IMAGE}"
[ -n "${NAS_MOUNT_PATH:-}" ] && OVERRIDES[nas-mount-path]="${NAS_MOUNT_PATH}"
[ -n "${ESM_CHECKPOINT:-}" ] && OVERRIDES[esm-checkpoint]="${ESM_CHECKPOINT}"
[ -n "${CONFIG_PATH:-}" ] && OVERRIDES[config-path]="${CONFIG_PATH}"

echo "=== DLP-Affinity train submit ==="
echo "template:   ${TEMPLATE_NAME}"
echo "namespace:  ${NAMESPACE}"
echo "exp-name:   ${EXP_NAME}"
echo "num-epochs: ${NUM_EPOCHS}"
echo "best-model: ${OUTPUT_DIR}/${EXP_NAME}/best_model.pt"

ARGS=(--from "workflowtemplate/${TEMPLATE_NAME}" -n "${NAMESPACE}")
for key in "${!OVERRIDES[@]}"; do
  ARGS+=(-p "${key}=${OVERRIDES[$key]}")
done

argo submit "${ARGS[@]}" "$@"
echo "submit done."
echo "  status: argo get -n ${NAMESPACE} @latest"
echo "  logs:   argo logs -n ${NAMESPACE} @latest | rg 'INFO -|ERROR|Training completed'"
