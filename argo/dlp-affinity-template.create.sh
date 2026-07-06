#!/usr/bin/env bash
# DLP-Affinity WorkflowTemplate 创建/更新脚本
# 输入：NAMESPACE（可选）；输出：在集群中注册 train / predict 两个 WorkflowTemplate
# 处理逻辑：kubectl apply 模板 YAML（幂等，存在则更新）
#
# 用法：
#   ./argo/dlp-affinity-template.create.sh           # 注册全部模板
#   ./argo/dlp-affinity-template.create.sh train     # 仅训练模板
#   ./argo/dlp-affinity-template.create.sh predict   # 仅预测模板
#   NAMESPACE=default ./argo/dlp-affinity-template.create.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"

TRAIN_TEMPLATE="${SCRIPT_DIR}/dlp-affinity-train.yaml"
PREDICT_TEMPLATE="${SCRIPT_DIR}/dlp-affinity-predict.yaml"

apply_one() {
  local name="$1"
  local file="$2"
  if [ ! -f "${file}" ]; then
    echo "ERROR: template file not found: ${file}" >&2
    exit 1
  fi
  echo "apply WorkflowTemplate/${name} (ns=${NAMESPACE})"
  kubectl apply -n "${NAMESPACE}" -f "${file}"
}

TARGET="${1:-all}"

echo "=== DLP-Affinity WorkflowTemplate create/update ==="
echo "namespace: ${NAMESPACE}"
echo "target:    ${TARGET}"

case "${TARGET}" in
  all)
    apply_one "dlp-affinity-train" "${TRAIN_TEMPLATE}"
    apply_one "dlp-affinity-predict" "${PREDICT_TEMPLATE}"
    ;;
  train)
    apply_one "dlp-affinity-train" "${TRAIN_TEMPLATE}"
    ;;
  predict)
    apply_one "dlp-affinity-predict" "${PREDICT_TEMPLATE}"
    ;;
  *)
    echo "Usage: $0 [all|train|predict]" >&2
    exit 1
    ;;
esac

echo "done. list: kubectl get workflowtemplate -n ${NAMESPACE} | grep dlp-affinity"
echo "  train:   argo submit --from workflowtemplate/dlp-affinity-train -n ${NAMESPACE}"
echo "  predict: argo submit --from workflowtemplate/dlp-affinity-predict -n ${NAMESPACE}"
echo "  or:      ./argo/dlp-affinity-train.submit.sh / ./argo/dlp-affinity-predict.submit.sh"
