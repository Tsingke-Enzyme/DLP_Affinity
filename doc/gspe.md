# GSPE（全局随机投影嵌入）方法详解

本文说明 DLP-Affinity 任务头中 **GSPE** 模块的设计目标、几何直觉、前向计算、扩展版统计/注意力特征、配置项与调参建议。代码：`release_package/models/gspe.py`。

总训练语境见 [训练指南](./training-guide.md)。R2R 见 [r2r.md](./r2r.md)。

---

## 1. 要解决什么问题

R2R 聚焦**残基对软对齐**路径。互补地，GSPE（Global Stochastic Projection Embedding，全局随机投影嵌入）从**整条链的集合几何**出发：把长度可变的残基集合映到与长度无关的摘要，再比较抗体集合与抗原集合的差异，得到固定维全局特征 \(h_{\mathrm{GSPE}}\)。

直觉：  
在高维空间里，一条蛋白链是一堆点云。随机扔出若干方向，把点云投影到这些方向上、排序后做聚合，得到一组「从该方向看过去的统计剪影」。若两条链整体分布不同，多组剪影之间的距离就会反映**全局形状差异**——与结合界面局部细节（R2R）形成双通路。

> 段末注释：Stochastic/随机指投影矩阵按高斯抽样初始化；默认投影方向**冻结**，不靠梯度学习旋转，借鉴随机投影稳定近邻几何的思想。

---

## 2. 在全模型中的位置

```text
X_ab ──┐
       ├─► GSPE ──► h_GSPE (num_groups 维，扩展版更长) ─┐
X_ag ──┘                                               ├─► concat ─► MLP ─► ŷ
                                           h_R2R ──────┘
```

| 输入 | 形状 |
|------|------|
| \(X_{ab}\) | \((L_{ab}, d)\) |
| \(X_{ag}\) | \((L_{ag}, d)\) |

| 输出版本 | 默认形状 |
|----------|----------|
| `GSPEModule` | \((\texttt{num\_groups},)\)，默认 8 维 |
| `GSPEModuleExtended` | `num_groups + 4·num_groups (+ attn)` |

`DLPAffinity` 中默认使用基础 `GSPEModule`（`gspe.use_extended=false`）。

---

## 3. 基础模块：`GSPEModule`

### 3.1 构造：多组单位投影

超参（`GSPEConfig`）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `num_projections` \(P\) | 64 | 每组内投影方向数 |
| `num_groups` \(G\) | 8 | 独立投影矩阵组数；输出维 |
| `trainable_projections` | false | `True` 时 \(R\) 可学习 |
| `aggregation` | `mean` | 对排序后投影坐标的聚合：`mean`/`max`/`sum` |

对每个组 \(g=1,\ldots,G\)，采样

\[
R^{(g)}\in\mathbb{R}^{P\times d},
\quad
\text{每行按 } L_2\text{ 归一化为单位向量}.
\]

- `trainable_projections=false`（默认）：`register_buffer` 存 \(R\)，**不报名优化器**。  
- `true`：`nn.Parameter`，可端到端调方向。

另有可学习尺度

\[
\sigma\in\mathbb{R}^{P},\quad \text{初始化全 } 1,
\]

用于距离归一化（对多尺度差异更稳）。

### 3.2 对单条链：投影 → 排序 → 聚合

定义 \(X\in\mathbb{R}^{L\times d}\)（抗体或抗原）。对组内矩阵 \(R\in\mathbb{R}^{P\times d}\)：

\[
\mathbf{P} = R\, X^{\top}
\in\mathbb{R}^{P\times L}.
\]

第 \(p\) 行是所有残基在第 \(p\) 个方向上的标量投影。再**沿残基维排序**：

\[
\mathbf{P}^{\mathrm{sorted}} = \mathrm{sort}_{L}(\mathbf{P}).
\]

排序使摘要对残基排列顺序近似不敏感，强调**多集合的一维经验分布**，而不是序列编号顺序本身（结合位点局部顺序本由 R2R 照顾）。

按 `aggregation` 在残基维上汇总，得到长度为 \(P\) 的剪影向量 \(\hat{p}\)：

| aggregation | \(\hat{p}_p\) |
|-------------|---------------|
| `mean` | \(\mathrm{mean}_{L}\,\mathbf{P}^{\mathrm{sorted}}_{p,:}\) |
| `max` | \(\max_{L}\) |
| `sum` | \(\sum_{L}\) |

实现函数：`project_and_sort`。记抗体侧 \(\hat{p}_A\)，抗原侧 \(\hat{p}_B\)。

### 3.3 双侧距离 → 非负特征

\[
z^{(g)}
=
\mathrm{softplus}\!\left(
\sqrt{
\sum_{p=1}^{P}
\left(
\frac{\hat{p}_{A,p}-\hat{p}_{B,p}}{\sigma_p+\varepsilon}
\right)^{2}
+\varepsilon}
\right).
\]

即先对剪影做按维 \(\sigma\)-缩放的欧氏距离，再 **softplus** 保证特征为正、梯度平滑。

最终输出：

\[
h_{\mathrm{GSPE}}
=
\big(z^{(1)},\ldots,z^{(G)}\big)
\in\mathbb{R}^{G}.
\]

默认 \(G=8\)：八个「从不同随机坐标族看过去」的全局差异标量。

---

## 4. 几何与方法意图

1. **集合摘要**：排序 + 聚合弱化排布索引，把可变长 \(L\) 变成固定 \(P\) 维。  
2. **随机投影**：Johnson–Lindenstrauss 型直觉：足够多随机方向可保距离结构的粗信息；随机初始化、默认不训练，避免过参数化。  
3. **多组独立影机**：\(G\) 组 \(R^{(g)}\) 像 \(G\) 个不同「相机」，差异信号更稳妥。  
4. **与 R2R 互补**：R2R 学局部配对；GSPE 给**整体形态差异**的无归纳偏置特征，二者拼接后由 MLP 融合。

复杂度：每组约为 \(O(P·L·d)\)（矩阵乘主导），\(G\) 组线性倍增；相对标准 self-attention \(O(L^2 d)\) 在长链上通常可接受。

---

## 5. 扩展版：`GSPEModuleExtended`

`gspe.use_extended=true` 时启用。在基础 \(z^{(g)}\) 之外可追加：

### 5.1 统计差分（`use_statistics=true`）

对单侧剪影栈 \(H\in\mathbb{R}^{G\times P}\)（`get_single_representation`），算 \(H_{ab}-H_{ag}\)，再沿方向维聚合：

\[
\mathrm{mean},\ \mathrm{std},\ \min,\ \max
\quad\text{各得 }G\text{ 维，共 }4G\text{ 维}.
\]

### 5.2 Attention 摘要（`use_attention=true`）

把 \(H_{ab}, H_{ag}\) 视为长度 \(G\)、宽为 \(P\) 的序列，跑一次 `MultiheadAttention`，再对方向维均值，得额外 \(G\) 维。

总输出维：

\[
\texttt{output\_dim}
= G + \underbrace{4G}_{\text{stats}} + \underbrace{G}_{\text{attn（可选）}}.
\]

默认训练路径**不用**扩展版，先把基础 GSPE 跑稳再开。

---

## 6. 默认可学习部分

| 张量 | 默认可训练 | 注释 |
|------|------------|------|
| \(R^{(g)}\) | 否 | buffer |
| \(\sigma\) | **是** | \(P\) 维尺度 |
| Extended 里的 MHA 等 | 依配置 | 扩展路径 |

因此冻 ESM、基础 GSPE 时，本模块自由度很小：**主要靠固定随机投影几何 + 学尺度 +（通过回归头）外部使用方式**。若全局通路表达不足，可试 `trainable_projections=true` 或 `use_extended`。

---

## 7. 调参建议

| 参数 | 建议 |
|------|------|
| `num_projections` | 增到 128 可更细剪影，增算力与 \(\sigma\) 维；减少则更快、更粗 |
| `num_groups` | 增大输出维与随机机位数；与回归头输入维联动 |
| `trainable_projections` | 数据充足且基础 GSPE 饱和时再开；防过拟合监视 val |
| `aggregation` | 默认 `mean`；`max` 更强调极端投影响应 |
| `use_extended` | 需要更丰富全局描述时；注意 `output_dim` 与回归头入口一致（由模型构造自动算） |

入口维 `input_dim` 必须等于 ESM `hidden_dim`（150M→640）。

---

## 8. 实现边界

1. **单对序列**：无 batch 维；batch 在 `DLPAffinity` 外循环。  
2. **无长度 mask**：pad 残基会进投影；当前路径多用真实长度嵌入。  
3. Extended 构造中额外复制了 `sigma` 字段，主路径距离用的是内部 `self.gspe.sigma`；读代码时注意别混淆。

---

## 9. 小结对照：R2R vs GSPE

| | R2R | GSPE |
|--|-----|------|
| 视角 | 残基对软对齐 | 点云全局投影差异 |
| 主技术 | KAN 压缩 + attention | 随机投影 + sort + 距离 |
| 默认输出维 | 32 | 8 |
| 参数量 | 较大（KAN） | 很小（默认仅 \(\sigma\)） |
| 归纳偏置 | 可学局部匹配 | 随机几何先验 |

二者拼接正是「局部交互 + 全局形态」双描述策略。

---

## 参考实现

```34:54:release_package/models/gspe.py
    def project_and_sort(self, X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        P = torch.matmul(R, X.T)
        P_sorted, _ = torch.sort(P, dim=1)
        # aggregation: mean / max / sum ...

    def forward(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        z_list = []
        for i in range(self.num_groups):
            R_i = self.R[i]
            hat_p_A = self.project_and_sort(X_ab, R_i)
            hat_p_B = self.project_and_sort(X_ag, R_i)
            z_list.append(F.softplus(self.compute_distance(hat_p_A, hat_p_B)))
        return torch.stack(z_list)
```
