# DLP-Affinity

基于深度学习的抗体–抗原结合亲和力（\(K_D\)）预测模型。以 ESM2 蛋白质语言模型为序列编码器，经 R2R（残基交互）与 GSPE（全局随机投影）提取配对特征，由回归头输出 \(\log_{10} K_D\)。

## 模型结构

```
抗体序列 + 抗原序列
        │
        ▼
   ESM2 Encoder（可冻结 / 可微调）
        │
   ┌────┴────┐
   ▼         ▼
  R2R      GSPE
   └────┬────┘
        ▼
  Regression Head（MLP）
        ▼
   predicted K_D (log10)
```

| 模块 | 作用 |
|------|------|
| ESM2 | 序列残基级表征 |
| R2R | 抗体–抗原残基对非线性交互（KAN + attention） |
| GSPE | 全局序列统计特征（随机投影） |
| Regression Head | 标量亲和力回归 |

方法细节见 [`doc/`](doc/README.md)。

## 仓库结构

```
DLP_Affinity/
├── release_package/          # 核心代码（训练 / 预测 / 模型 / 数据）
│   ├── train.py              # 亲和力微调入口
│   ├── predict.py            # 推理入口
│   ├── configs/config.py     # 默认超参与配置类
│   ├── models/               # ESM2、R2R、GSPE、DLPAffinity
│   └── data/                 # 数据集与样例 CSV
├── argo/                     # 阿里云 Argo 部署（WorkflowTemplate + 投递脚本）
│   ├── dlp-affinity-train.yaml
│   ├── dlp-affinity-predict.yaml
│   ├── dlp-affinity-template.create.sh   # 注册模板
│   ├── dlp-affinity-train.submit.sh      # 提交训练
│   ├── dlp-affinity-predict.submit.sh    # 提交预测
│   └── dlp-affinity-image.build.sh       # 构建并推送 Docker 镜像
├── doc/                      # 训练指南与 R2R / GSPE 详解
├── Dockerfile                # 运行镜像定义
└── dev_log/                  # 开发过程记录
```

## 环境要求

- Python 3.10+
- PyTorch 2.3.x + CUDA（集群镜像已内置）
- `transformers==4.46.3`（与 torch 2.3.1 兼容，勿随意升级）

本地安装：

```bash
cd release_package
pip install -r requirements.txt
```

## 数据格式

训练 / 预测 CSV 需包含抗体与抗原序列列，支持以下列名别名：

| 标准列名 | 别名 |
|----------|------|
| `seq_ab` | `antibody_seq` |
| `seq_ag` | `antigen_seq` |
| `kd`（训练） | `escape_fraction`（自动 log 变换） |

样例数据：`release_package/data/7KMG/LY-CoV555_DMS_*_model_input.csv`。

## 本地使用

### 训练

```bash
cd release_package

python train.py \
  --train_path data/7KMG/LY-CoV555_DMS_train_model_input.csv \
  --val_path data/7KMG/LY-CoV555_DMS_val_model_input.csv \
  --esm_model /path/to/esm2_t30_150M_UR50D \
  --output_dir ./outputs \
  --exp_name my_exp \
  --num_epochs 50 \
  --freeze_esm \
  --device cuda
```

产物：`outputs/<exp_name>/best_model.pt`（验证集最优）、`epoch_*.pt`（每轮 checkpoint）。

### 预测

```bash
python predict.py \
  --checkpoint ./outputs/my_exp/best_model.pt \
  --input data/7KMG/LY-CoV555_DMS_val_model_input.csv \
  --output ./outputs/predictions.csv \
  --device cuda
```

输出列：`predicted_kd_log10`、`predicted_kd`。

## 阿里云 Argo 部署

集群侧约定：

- NAS 挂载：`pvc-nas` → `/mnt`
- 项目根目录：`/mnt/nas1/liubo/project/DLP_Affinity`
- ESM2 基座：`/mnt/nas1/liubo/models/esm2_t30_150M_UR50D`
- GPU：ACS A10，`nvidia.com/gpu: 1`

### 1. 构建镜像（可选，代码变更或依赖更新时）

```bash
./argo/dlp-affinity-image.build.sh
```

### 2. 注册 WorkflowTemplate

```bash
./argo/dlp-affinity-template.create.sh        # train + predict
./argo/dlp-affinity-template.create.sh train  # 仅训练
```

模板为**单层 entrypoint**设计：单任务直接运行 `train.py` / `predict.py`，无多余调度嵌套。

### 3. 提交训练

```bash
./argo/dlp-affinity-train.submit.sh

# 自定义 epoch / 实验名
NUM_EPOCHS=20 EXP_NAME=exp001 ./argo/dlp-affinity-train.submit.sh

# 或通过 Argo 参数覆盖
./argo/dlp-affinity-train.submit.sh -p num-epochs=10
```

训练产物：`/mnt/nas1/liubo/project/DLP_Affinity/outputs/<exp_name>/best_model.pt`。

可将最新实验软链到固定路径，便于预测默认加载：

```bash
ln -sfn dlp-affinity-train_20260707_105354 /mnt/nas1/liubo/project/DLP_Affinity/outputs/dlp-affinity-train
```

### 4. 提交预测

```bash
# 默认读取 outputs/dlp-affinity-train/best_model.pt（可用软链指向最新实验）
./argo/dlp-affinity-predict.submit.sh

# 或显式指定 checkpoint
EXP_NAME=dlp-affinity-train_20260707_105354 ./argo/dlp-affinity-predict.submit.sh
```

### 5. 查看状态与日志

```bash
argo get -n default @latest
argo logs -n default @latest | rg 'INFO -|ERROR|Training completed'

# 或读 NAS 落盘日志（推荐，避免 argo watch 日志流过大断开）
tail -f /mnt/nas1/liubo/project/DLP_Affinity/outputs/<exp_name>/argo_main.log
```

## 主要可调参数

| 参数 | 位置 | 默认 | 说明 |
|------|------|------|------|
| `num-epochs` / `--num_epochs` | Argo / CLI | 50 | 训练轮数 |
| `freeze-esm` / `--freeze_esm` | Argo / CLI | true | 冻结 ESM2，仅训练任务头 |
| `esm-model-path` / `--esm_model` | Argo / CLI | NAS 150M | ESM2 本地目录或 HF id |
| `exp-name` / `--exp_name` | Argo / CLI | 带时间戳 | 实验输出子目录名 |
| `batch_size` | config | 8 | 批大小 |
| `learning_rate` | config | 1e-4 | AdamW 学习率 |
| `early_stopping_patience` | config | 10 | 验证集无改善则早停 |

完整参数与训练原理见 [`doc/training-guide.md`](doc/training-guide.md)。

## 文档索引

| 文档 | 内容 |
|------|------|
| [`doc/training-guide.md`](doc/training-guide.md) | 训练流程、损失函数、早停、Argo 对应关系 |
| [`doc/r2r.md`](doc/r2r.md) | R2R 模块设计 |
| [`doc/gspe.md`](doc/gspe.md) | GSPE 模块设计 |
| [`release_package/README.md`](release_package/README.md) | 代码包内 API 与配置说明 |

## 常见问题

**预测报 `checkpoint not found`**  
检查 `exp-name` 与训练输出目录是否一致，或对 `outputs/dlp-affinity-train` 建软链。

**预测报 `Unexpected key(s): esm_encoder._model.*`**  
确保 NAS 上的 `release_package/predict.py` 已同步（加载权重前需 eager-load ESM）。

**训练报 `transformers` 找不到 PyTorch**  
镜像内须保持 `transformers==4.46.3`；≥4.52 要求 torch≥2.4，会与基础镜像 2.3.1 冲突。

**Argo 优先使用 NAS 代码**  
Pod 内若存在 `${WORK_ROOT}/release_package/train.py`，会覆盖镜像内 `/app` 代码；hotfix 后需 rsync 到 NAS。
