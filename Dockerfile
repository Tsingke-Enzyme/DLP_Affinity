# DLP-Affinity 运行环境镜像
# 基础镜像与 pip 均使用国内源；与 Enzyme_designer 一致使用 biocloud PyTorch
# 构建/推送：./argo/dlp-affinity-image.build.sh → enzyme_dev:DLP_Affinity.v1.0.1
# 代码目录：/app；业务数据与 checkpoint 由 Argo NAS（pvc-nas -> /mnt）注入

ARG BASE_IMAGE=beijing-acr-cr-registry-vpc.cn-beijing.cr.aliyuncs.com/biocloud/pytorch:2.3.1-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="dlp-affinity"
LABEL org.opencontainers.image.description="DLP-Affinity antibody-antigen binding affinity train/predict"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    # HuggingFace 权重国内镜像，避免直连 huggingface.co
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONPATH=/app \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# apt 使用阿里云镜像；安装最小系统依赖
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i \
        -e 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' \
        -e 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' \
        -e 's|https://archive.ubuntu.com|https://mirrors.aliyun.com|g' \
        -e 's|https://security.ubuntu.com|https://mirrors.aliyun.com|g' \
        /etc/apt/sources.list; \
    fi; \
    if [ -d /etc/apt/sources.list.d ]; then \
      find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -print0 \
        | xargs -0 -r sed -i \
          -e 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' \
          -e 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' \
          -e 's|https://archive.ubuntu.com|https://mirrors.aliyun.com|g' \
          -e 's|https://security.ubuntu.com|https://mirrors.aliyun.com|g'; \
    fi; \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 仅复制代码与依赖清单；训练数据 / checkpoint 由 NAS 注入，避免镜像膨胀
COPY release_package/requirements.txt /app/requirements.txt
COPY release_package/ /app/

# Python 依赖走阿里云 PyPI
# transformers 须钉在 4.46.x：>=4.52 会因要求 torch>=2.4 而禁用现有 2.3.1 后端
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# 打开时 workingDir=/app，避免 NAS 挂载到 /mnt 遮蔽 PyTorch 运行时
WORKDIR /app

ENTRYPOINT []
CMD ["python", "train.py", "--help"]
