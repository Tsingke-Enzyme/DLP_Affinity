#!/usr/bin/env bash
# DLP-Affinity 镜像构建与推送
# 输入：可选环境变量覆盖镜像名/tag/基础镜像；输出：推送后的 IMAGE
# 处理逻辑：linux/amd64 单 manifest 构建 → 校验 transformers → push（公网/VPC ACR）
#
# 用法：
#   # 外网 Mac：先 login 公网 ACR，再构建推送（脚本会同时打 VPC tag 供集群使用）
#   docker login beijing-acr-cr-registry.cn-beijing.cr.aliyuncs.com
#   ./argo/dlp-affinity-image.build.sh
#   PUSH=false ./argo/dlp-affinity-image.build.sh   # 仅本地构建
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 集群内使用 VPC 域名；本机构建基础镜像走公网 registry
IMAGE_REPO_VPC="${IMAGE_REPO_VPC:-beijing-acr-cr-registry-vpc.cn-beijing.cr.aliyuncs.com/biocloud/enzyme_dev}"
IMAGE_REPO_PUBLIC="${IMAGE_REPO_PUBLIC:-beijing-acr-cr-registry.cn-beijing.cr.aliyuncs.com/biocloud/enzyme_dev}"
IMAGE_TAG="${IMAGE_TAG:-DLP_Affinity.v1.0.0}"
IMAGE_VPC="${IMAGE_VPC:-${IMAGE_REPO_VPC}:${IMAGE_TAG}}"
IMAGE_PUBLIC="${IMAGE_PUBLIC:-${IMAGE_REPO_PUBLIC}:${IMAGE_TAG}}"
# Mac 上常连不到 VPC ACR，默认用已有公网同源 pytorch 基座
BASE_IMAGE="${BASE_IMAGE:-registry.cn-zhangjiakou.aliyuncs.com/biocloud/pytorch:2.3.1-cuda12.1-cudnn8-devel}"
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH="${PUSH:-true}"

echo "=== DLP-Affinity image build ==="
echo "context:      ${ROOT_DIR}"
echo "image_vpc:    ${IMAGE_VPC}"
echo "image_public: ${IMAGE_PUBLIC}"
echo "base_image:   ${BASE_IMAGE}"
echo "platform:     ${PLATFORM}"
echo "push:         ${PUSH}"

# 关闭 provenance/sbom，避免 ACR 对 multi-manifest attestation 推送失败
docker buildx build \
  --platform "${PLATFORM}" \
  --provenance=false \
  --sbom=false \
  --load \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE_VPC}" \
  -t "${IMAGE_PUBLIC}" \
  -f "${ROOT_DIR}/Dockerfile" \
  "${ROOT_DIR}"

# 校验 transformers 已钉为 4.46.3，避免再出现 torch 2.3.1 被禁用
echo "=== verify transformers pin ==="
docker run --rm --platform "${PLATFORM}" --entrypoint python "${IMAGE_VPC}" -c \
  "import importlib.metadata as m, torch; from transformers import AutoModel; \
print('torch', torch.__version__); \
print('transformers', m.version('transformers')); \
assert m.version('transformers') == '4.46.3', m.version('transformers')"

if [ "${PUSH}" != "true" ]; then
  echo "skip push (PUSH=false); local tags ready: ${IMAGE_VPC}"
  exit 0
fi

push_one() {
  local ref="$1"
  echo "=== push ${ref} ==="
  if docker push "${ref}"; then
    echo "push done: ${ref}"
    return 0
  fi
  return 1
}

# 优先推公网域名（外网可达）；同实例下集群可用 VPC 域名拉取同 tag
if push_one "${IMAGE_PUBLIC}"; then
  # 再尝试 VPC 域名（内网机构建机可成功；Mac 上通常失败，可忽略）
  push_one "${IMAGE_VPC}" || echo "warn: VPC endpoint push failed (expected off-VPC); cluster pulls ${IMAGE_VPC} after public push"
  exit 0
fi

echo "ERROR: push failed. Login ACR then retry:" >&2
echo "  docker login beijing-acr-cr-registry.cn-beijing.cr.aliyuncs.com" >&2
echo "  ./argo/dlp-affinity-image.build.sh" >&2
exit 1
