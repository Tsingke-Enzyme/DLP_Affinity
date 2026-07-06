# DLP-Affinity 微调训练指南

本文整合本项目的训练方法、原理、任务头组成与可调参数，面向微调入门读者：专业表述与通俗解释并存。实现细节以 `release_package/` 代码为准。

---

## 1. 要解决什么问题

给定**抗体氨基酸序列**与**抗原氨基酸序列**，模型输出与结合强弱相关的连续数值。

论文语境里常见标签为结合亲和力常数（equilibrium dissociation constant，\(K_D\)）。当前 Argo / 7KMG DMS 流水线使用逃逸分数（escape fraction）列 `escape_fraction`，在代码中与 `kd` 接口共用同一回归流程；默认对标签做 \(\log_{10}\) 变换，降低数值跨度。

> 段末注释：\(K_D\) 越小通常表示结合越强；DMS 的 escape_fraction 语义与 \(K_D\) 不同，但本项目训练管线将其当作回归标签处理，评估时需与业务语义对齐。

一条训练样本的逻辑结构：

| 字段 | 含义 | 7KMG CSV 列名示例 |
|------|------|-------------------|
| 抗体序列 | 氨基酸字符串 | `antibody_seq` |
| 抗原序列 | 氨基酸字符串 | `antigen_seq` |
| 标签 | 连续目标（默认 log 空间学习） | `escape_fraction` |

推理时输入只需两条序列，模型给出 \(\hat{y}\)（默认在 log 空间），再按需反变换。

---

## 2. 训练流水线总览

```text
通用 ESM2 预训练权重
        │
        ├─（可选）抗体相关 MLM 继续适应 ──► esm_checkpoint (.pt)
        │         train_mlm.py
        ▼
亲和力 / 标签回归训练（train.py）  ◄── 当前主路径
        │
        ▼
best_model.pt（验证集最优快照）
        │
        ▼
predict.py 预测
```

**ESM2**（Evolutionary Scale Modeling 2，进化尺度建模第二代）是蛋白语言模型，在海量蛋白序列上预训练，擅长把序列变成残基质向量（residue embedding）。它本身不直接输出亲和力数值。

> 段末注释：ESM2 由 Meta 开源；本项目经 HuggingFace `transformers` 的 `AutoModel` 加载本地 HF 目录或模型 id。

本项目在 ESM2 之上接**任务相关头网络**，把一对（抗体、抗原）表示压成一个标量。当前集群任务默认更轻量：

| 项 | 当前默认 |
|----|----------|
| 基座 | NAS 上 `esm2_t30_150M_UR50D`（hidden_dim=640） |
| ESM 是否更新 | **冻结**（`--freeze_esm`） |
| 实际学习对象 | R2R + GSPE + Regression Head |
| Epochs | **50**，无早停 |
| 选模策略 | 验证集 loss 最低时保存 `best_model.pt` |

这属于**冻结骨干 + 下游头微调**：ESM2 当固定特征提取器，只学「如何从这些特征得到标签」。

可选升级路径：

| 档位 | ESM | 场景 |
|------|-----|------|
| A. 冻骨干 | 不更新 | 数据少、省显存、baseline（当前） |
| B. 解冻末几层 | `unfreeze_last_n_layers` | 适度适配 |
| C. 全量微调 | `freeze_esm=false` | 数据多；学习率通常更小 |
| D. 先 MLM 再回归 | `--esm_checkpoint` | 有抗体域序列时域适应 |

> 段末注释：MLM（Masked Language Modeling，掩码语言建模）是「挖洞填词」式预训练目标；亲和力阶段本身不跑 MLM。

---

## 3. 模型总架构

实现类：`models/dlp_affinity.py` 中的 `DLPAffinity`。整条前向：

```text
抗体序列 ──┐
           ├─► ESM2 编码器 ─► 残基嵌入 X_ab, X_ag
抗原序列 ──┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
         R2R                 GSPE
   （局部残基对关系）     （全局随机投影摘要）
          │                   │
          └────────┬──────────┘
                   ▼
            拼接特征向量 h
                   ▼
         AffinityRegressor（MLP）
                   ▼
              预测标量 ŷ
```

记抗体、抗原残基序列长分别为 \(L_{ab}\)、\(L_{ag}\)，ESM 隐维度为 \(d\)（150M 时 \(d=640\)），则

\[
X_{ab}\in\mathbb{R}^{L_{ab}\times d},\quad
X_{ag}\in\mathbb{R}^{L_{ag}\times d}.
\]

任务头输出：

\[
\hat{y} = f_{\mathrm{MLP}}\big(
  [\,h_{\mathrm{R2R}};\, h_{\mathrm{GSPE}}\,]
\big).
\]

框架均为 **PyTorch**（`torch.nn`）；ESM2 经 **HuggingFace Transformers** 加载；优化器为训练循环内手写 **AdamW**，不依赖 Lightning / HF Trainer。

任务头三块详解见：

- [R2R 方法详解](./r2r.md)
- [GSPE 方法详解](./gspe.md)

回归头是标准多层感知机（multilayer perceptron，MLP）：`Linear → ReLU → Dropout`，最后一层映射到 1 维。

> 段末注释：MLP 即多层全连接网络，本项目里的 `AffinityRegressor`。

---

## 4. 数据如何进入训练

`data/dataset.py` 中 `AffinityDataset`：

1. 读 CSV / JSON，解析抗体、抗原、标签列（支持别名，如 DMS 列名）。
2. 按 `max_ab_len` / `max_ag_len` 过滤过长序列。
3. 可选 \(\log_{10}\) 变换标签。
4. 可选按标签频次分层采样（`use_stratified_sampler`），缓解标签桶不平衡。

训练集：用于反向传播更新参数。  
验证集：每个 epoch 结束后评估，**不参与梯度**；用于挑选 `best_model.pt`。

只看训练 loss 容易「背题」（过拟合）；验证集指标才反映泛化。

---

## 5. 训练过程与原理

### 5.1 一个 step 在干什么

一个 **batch**（批次，默认大小 8）上固定四步：

1. **前向**：序列 → ESM（冻结时 `requires_grad=False`）→ R2R / GSPE / MLP → \(\hat{y}\)
2. **算损失**：默认均方误差（mean squared error，MSE）

\[
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
\]

配置里还可开 Huber、相关损失等（默认多半关闭）。

3. **反向**：\(\partial L / \partial\theta\)，\(\theta\) 为可训练参数（冻 ESM 时主要是任务头）。
4. **更新**：AdamW；学习率经 **warmup + 线性衰减**。

> 段末注释：MSE 即平方误差平均；warmup 指训练初期把学习率从较低值升到目标值，减轻初期不稳。

### 5.2 Epoch：顺序迭代，不是并行多次训练

一个 **epoch**（历元）= 训练集按 batch 完整扫一遍。当前数据规模下约 **195** step / epoch，默认共 **50** 个 epoch。

参数更新链：

\[
\theta_0 \xrightarrow{\text{epoch 1}} \theta_1 \xrightarrow{\text{epoch 2}} \cdots \xrightarrow{\text{epoch 50}} \theta_{50}.
\]

这是**同一条训练轨迹上的串行依赖**，不是并行起 50 个独立实验再挑最好。每一轮都会改参数；batch 之间也是顺序 `backward → step`。

每轮另外做：

| 动作 | 含义 |
|------|------|
| 记录 train loss | 拟合程度 |
| 验证 val loss / correlation / RMSE | 泛化打分 |
| val loss 创新低 → 覆盖 `best_model.pt` | 保留验证最优快照 |
| 保存 `epoch_n.pt` | 轨迹上第 \(n\) 轮整模状态（含优化器） |

### 5.3 若 best_model 来自中间某轮是什么意思

常见曲线：前期 train / val 同降；中后期 train 还降而 val 变差（过拟合）。

若最终 `best_model` 是第 8 轮：

- **对验证指标**：第 9–50 轮没有更好，甚至更差，所以部署用第 8 轮权重。
- **对参数演化**：后面轮次仍在更新，多半进一步拟合训练集，不等于「什么都没算」。

没有早停（early stopping）时会把 50 轮跑满；只是推理推荐读 `best_model.pt`。

> 段末注释：早停指验证指标若干轮不提升就停止训练，本项目 `train.py` **尚未实现**。

### 5.4 与优化数学的直觉对应

可训练部分最小化训练损：

\[
\theta^\star_{\mathrm{train}}
\approx
\arg\min_\theta
\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{train}}}\big[L(f_\theta(x), y)\big],
\]

但我们真正关心的是验证（或测试）期望。用验证 loss 选模，是对泛化风险的廉价代理。

---

## 6. 如何读日志指标

| 指标 | 含义 |
|------|------|
| Train Loss | 训练拟合；持续下降常见 |
| Val Loss | **当前写 `best_model` 的主判据** |
| Correlation | 预测排序与真实排序一致性 |
| RMSE / MAE | 整体偏差 |

排亲和力相关任务时，除了 val loss，也应看 correlation 是否同步改善。

---

## 7. 可调参数一览（后续实验优先序）

配置定义于 `configs/config.py`；Argo 已透出部分：`--freeze_esm`、`--esm_model`、数据路径等。细粒度改动写 JSON config，或改默认后同步 **NAS** 上的 `release_package`（Pod **优先跑 NAS 代码**）。

### 7.1 优先调（收益/可控性强）

| 参数 | 默认约值 | 效果 |
|------|----------|------|
| `num_epochs` | 50 | 训练多久；建议配合早停 |
| `learning_rate` | \(1\times 10^{-4}\) | 过大震荡，过小偏慢；解冻 ESM 时常降到 \(10^{-5}\) 量级 |
| `batch_size` | 8 | 显存与梯度噪声 |
| `freeze_esm` | true（当前任务） | 是否更新骨干 |
| `warmup_steps` | 100 | 前期热身步数 |
| `weight_decay` | 0.01 | 正则强度 |
| `seed` | 42 | 可复现；多 seed 评估稳健性 |

### 7.2 结构与损失

| 参数 | 作用 |
|------|------|
| `regressor.hidden_dims` / `dropout` | 回归头容量与正则 |
| `r2r.*` | 见 [r2r.md](./r2r.md) |
| `gspe.*` | 见 [gspe.md](./gspe.md) |
| `use_huber_loss` | 对异常点更稳 |
| `use_correlation_loss` | 直接奖励相关排序 |

### 7.3 数据与骨干

| 参数 | 注意 |
|------|------|
| `log_transform_kd` | 须与评估尺度一致 |
| `max_ab_len` / `max_ag_len` | 截断过短丢信息 |
| `esm.model_name` / `--esm_model` | 规模与显存 |
| `--esm_checkpoint` | MLM 微调后的 `.pt` |
| `lora.enabled` | 参数高效微调路径（配置有预留） |

---

## 8. 与当前 Argo 任务的对应关系

| 项 | 现状 |
|----|------|
| 入口 | `argo/dlp-affinity-train.submit.sh` |
| 基座 | `/mnt/nas1/liubo/models/esm2_t30_150M_UR50D` |
| 代码优先级 | `${WORK_ROOT}/release_package`，否则镜像 `/app` |
| 设备 | `cuda`；资源 `nvidia.com/gpu: 1`，型号标签 A10 |
| 输出目录 | `${OUTPUT_DIR}/${EXP_NAME}/`，含 `best_model.pt`、`epoch_*.pt`、`argo_main.log` |

日志侧若见 pooler 权重「newly initialized」，在冻结 ESM、不接 pooler 训练时通常可忽略，只要 encoder 主干从本地 HF 目录加载成功即可。

---

## 9. 实践路线（给入门同学）

1. 读懂 log：何时 `Saved best model!`，val loss / correlation 曲线走向。
2. 确认最优出现在第几轮：很早最好且后续变差 → 过拟合倾向；考虑降容量、加 dropout/WD、早停。
3. 冻 ESM 时优先动：LR、epochs/早停、回归头 dropout。
4. 确认收益后再试解冻 ESM、更大 ESM 或 MLM checkpoint，并降低学习率。

一句话：**预训练 ESM 负责「读序列」，R2R/GSPE/MLP 负责「读出结合强弱」；训练在同一参数轨迹上迭代更新，并用验证集留下历史上最准的快照。**

---

## 参考代码路径

| 路径 | 内容 |
|------|------|
| `release_package/train.py` | 训练循环、选模 |
| `release_package/configs/config.py` | 全部默认超参 |
| `release_package/models/dlp_affinity.py` | 总模型与回归头 |
| `release_package/models/esm_encoder.py` | ESM2 编码 |
| `release_package/models/r2r.py` | R2R |
| `release_package/models/gspe.py` | GSPE |
| `release_package/data/dataset.py` | 数据加载 |
| `argo/dlp-affinity-train.yaml` | 集群投递模板 |
