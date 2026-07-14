#!/usr/bin/env bash
# DLP-Affinity WorkflowTemplate 注册/更新（单层模板，不含投递逻辑）
#
# 用法：
#   ./argo/dlp-affinity-template.create.sh                # 注册 train + predict + train_predict
#   ./argo/dlp-affinity-template.create.sh train
#   ./argo/dlp-affinity-template.create.sh predict
#   ./argo/dlp-affinity-template.create.sh train_predict
#   NAMESPACE=default ./argo/dlp-affinity-template.create.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-default}"
TARGET="${1:-all}"

apply_one() {
  local file="$1"
  if [ ! -f "${file}" ]; then
    echo "ERROR: template not found: ${file}" >&2
    exit 1
  fi
  echo "apply $(basename "${file}") (ns=${NAMESPACE})"
  kubectl apply -n "${NAMESPACE}" -f "${file}"
}

echo "=== DLP-Affinity WorkflowTemplate create/update ==="
echo "namespace: ${NAMESPACE}"
echo "target:    ${TARGET}"

case "${TARGET}" in
  all)
    apply_one "${SCRIPT_DIR}/dlp-affinity-train.yaml"
    apply_one "${SCRIPT_DIR}/dlp-affinity-predict.yaml"
    apply_one "${SCRIPT_DIR}/dlp-affinity-train_predict.yaml"
    ;;
  train)
    apply_one "${SCRIPT_DIR}/dlp-affinity-train.yaml"
    ;;
  predict)
    apply_one "${SCRIPT_DIR}/dlp-affinity-predict.yaml"
    ;;
  train_predict|train-predict)
    # 串联模板依赖 train/predict 的 templateRef，需先/同时注册
    apply_one "${SCRIPT_DIR}/dlp-affinity-train.yaml"
    apply_one "${SCRIPT_DIR}/dlp-affinity-predict.yaml"
    apply_one "${SCRIPT_DIR}/dlp-affinity-train_predict.yaml"
    ;;
  *)
    echo "Usage: $0 [all|train|predict|train_predict]" >&2
    exit 1
    ;;
esac

echo "done."
echo "  list:   kubectl get workflowtemplate -n ${NAMESPACE} | rg dlp-affinity"
echo "  train:  ./argo/dlp-affinity-train.submit.sh"
echo "  predict: ./argo/dlp-affinity-predict.submit.sh"
echo "  train_predict: ./argo/dlp-affinity-train_predict.submit.sh"
