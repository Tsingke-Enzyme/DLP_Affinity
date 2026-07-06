# R2R（残基对残基交互）方法详解

本文说明 DLP-Affinity 任务头中 **R2R** 模块的设计目标、前向计算、与 KAN 的关系、配置项与实现注意点。代码：`release_package/models/r2r.py`，依赖 `release_package/models/kan.py`。

总训练语境见 [训练指南](./training-guide.md)。

---

## 1. 要解决什么问题

ESM2 为抗体、抗原各自产出残基级嵌入 \(X_{ab}\)、\(X_{ag}\)，但**双侧独立编码并不显式建模跨链接触/匹配关系**。R2R（Residue-to-Residue，残基对残基）模块的目标是：

> 在抗体每个残基位置上，汇总与抗原各残基的可学习交互信息，再池化成固定长度向量 \(h_{\mathrm{R2R}}\)，供回归头使用。

直觉类比：先把每个残基压成「细指纹」，再做抗体侧对整个抗原侧的 **attention（注意力）软对齐**，让抗体上每个位置带着「它最关注到哪些抗原位点」的上下文，经非线性混合后聚合成一条交互描述。

> 段末注释：Attention/注意力在本文中指对抗原维度做 softmax 归一化的权重分配，不是完整 Transformer 块。

---

## 2. 在全模型中的位置

```text
X_ab (L_ab × d) ──┐
                  ├─► R2R ──► h_R2R (r2r_out_dim,)  ─┐
X_ag (L_ag × d) ──┘                                  ├─► concat ─► MLP ─► ŷ
                                                     │
                                           h_GSPE ───┘
```

输入：

| 符号 | 形状 | 含义 |
|------|------|------|
| \(X_{ab}\) | \((L_{ab}, d)\) | 抗体残基嵌入，\(d\) 为 ESM hidden_dim（150M 时 640） |
| \(X_{ag}\) | \((L_{ag}, d)\) | 抗原残基嵌入 |

输出：

| 符号 | 形状 | 含义 |
|------|------|------|
| \(h_{\mathrm{R2R}}\) | \((d_{\mathrm{r2r}},)\) | 默认 \(d_{\mathrm{r2r}}=32\)，一对序列一条向量 |

实现类：默认 `R2RModule`；`r2r.use_simple=true` 时用 `R2RModuleSimple`（交互侧改用 MLP）。

---

## 3. 标准路径：`R2RModule` 分步计算

默认配置下主要超参（`configs/config.py` → `R2RConfig`）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `compress_dim` | 1 | 每侧 KAN 压缩后的通道维（实现里作交互用指纹维） |
| `output_dim` (`r2r_out_dim`) | 32 | 最终池化前每条残基特征维 / 池化后输出维 |
| `kan_hidden_dims_reduce` | `[512,128,32]` | 压缩用 KAN 宽度 |
| `kan_hidden_dims_inter` | `[256,128]` | 交互融合用 KAN 宽度 |
| `num_knots` | 8 | KAN 样条基结点数标度 |
| `pooling` | `mean` | `mean` / `max` / `attention` |

记 \(c=\)`compress_dim`，\(d_o=\)`r2r_out_dim`。

### 步骤 1：双侧残基压缩（KANReduce）

对每一条残基嵌入做可学习非线性降维：

\[
V_{ab} = \mathrm{KANReduce}_{ab}(X_{ab}) \in \mathbb{R}^{L_{ab}\times c},\quad
V_{ag} = \mathrm{KANReduce}_{ag}(X_{ag}) \in \mathbb{R}^{L_{ag}\times c}.
\]

默认 \(c=1\) 时，每个残基变成一个标量指纹，便于构造 \(L_{ab}\times L_{ag}\) 的相似度矩阵。两侧各有一套 `kan_ab` / `kan_ag`，参数不共享。

**KAN**（Kolmogorov–Arnold Network，柯尔莫哥洛夫–阿诺德网络）在本项目中是自实现的层：在基路径（SiLU + 线性）之外，对每个输入维叠加一组可学习径向基状「样条」系数，表达比单层 ReLU-MLP 更丰富的一元非线性。堆叠为 `KAN` / `KANReduce`。

> 段末注释：此处 KAN 是论文式启发下的工程实现，细节见 `models/kan.py` 的 `KANLayer`（基函数非严格 B-样条，实现为高斯宽基）。

### 步骤 2：残基对分数矩阵

\[
M_{\mathrm{inter}} = V_{ab}\, V_{ag}^{\top}
\in \mathbb{R}^{L_{ab}\times L_{ag}}.
\]

入口 \((i,j)\) 可理解为：抗体第 \(i\) 位指纹与抗原第 \(j\) 位指纹的相容度（当 \(c>1\) 时为向量点积）。

### 步骤 3：对抗原维度做注意力

对每一抗体位置 \(i\)，在抗原位置上 softmax：

\[
A = \mathrm{softmax}_{j}\big(M_{\mathrm{inter}}\big)
\in \mathbb{R}^{L_{ab}\times L_{ag}}.
\]

抗原指纹经 `ag_proj: Linear(1, 64)` 升维（当 \(c=1\) 时输入是标量通道）：

\[
V'_{ag} = \mathrm{Linear}(V_{ag})\in\mathbb{R}^{L_{ag}\times 64}.
\]

加权求和得到抗体侧的「抗原语境」：

\[
M_{\mathrm{proj}} = A\, V'_{ag}
\in\mathbb{R}^{L_{ab}\times 64}.
\]

即：抗体每个残基带着一份软对齐后的抗原摘要。

### 步骤 4：与自身指纹拼接，再经交互 KAN

\[
M_{\mathrm{concat}} = \big[\,M_{\mathrm{proj}}\,;\, V_{ab}\,\big]
\in\mathbb{R}^{L_{ab}\times (64+c)},
\]

\[
H_{AA} = \mathrm{KAN}_{\mathrm{inter}}(M_{\mathrm{concat}})
\in\mathbb{R}^{L_{ab}\times d_o}.
\]

每个抗体残基位置得到 \(d_o\) 维交互特征。

### 步骤 5：池化成单向量

| `pooling` | 行为 |
|-----------|------|
| `mean`（默认） | \(h_{\mathrm{R2R}} = \mathrm{mean}_{i} H_{AA}[i]\) |
| `max` | 各维取 \(\max_i\) |
| `attention` | 可学习打分 softmax 后加权和 |

输出供 `DLPAffinity.forward_single` 与 GSPE 向量拼接。

---

## 4. 为什么这样设计（方法意图）

1. **跨链显式交互**：矩阵 \(M_{\mathrm{inter}}\) 与 attention 让信息在抗体–抗原位置间流动，而不是只各做 global pool 再拼。
2. **压缩再交互**：\(d\)（几百～上千）直接做 \(L\times L\times d\) 全隔离交互代价大；先压到 \(c\) 维控制复杂度（默认 \(c=1\) 时类似「学会的一维打分器」）。
3. **KAN 而非纯 MLP**：压缩与融合路径用可学习激活基，强调残基特征维度上的曲线形非线性，契合项目方法叙事。
4. **输出固定维**：池化后与序列长度解耦，方便与 GSPE、回归头拼接。

复杂度量级：注意力部分约为 \(O(L_{ab} L_{ag})\)，在抗体/抗原几百长度级别可接受。

---

## 5. 简化版：`R2RModuleSimple`

`r2r.use_simple=true`（如 `get_small_config()`）时：

1. 同样 `KANReduce` 得 \(V_{ab}, V_{ag}\)。
2. \(M_{\mathrm{inter}}=V_{ab}V_{ag}^{\top}\)。
3. **直接** `[M_inter; V_ab]` 拼接到抗体维（此时 \(M_{\mathrm{inter}}\) 的行是长度为 \(L_{ag}\) 的评分），再过 **MLP**（SiLU + Dropout），不再走 `ag_proj` + `inter_kan`。

用途：联调、小配置冒烟；表达力通常弱于标准 `R2RModule`。

---

## 6. 可训练参数与冻结 ESM 时的角色

冻 ESM 时，R2R 是主要可学习交互通路之一，参数包括：

- `kan_ab` / `kan_ag` 全部权重
- `ag_proj`
- `inter_kan`（或 Simple 中的 `inter_mlp`）
- `pooling=attention` 时的 `attention_weight`

可选：`DLPAffinity.apply_lora` 可对 R2R 内 `KANLayer` 挂 LoRA（配置预留）。

---

## 7. 调参建议

| 参数 | 增大倾向 | 减小倾向 |
|------|----------|----------|
| `output_dim` | 交互描述更富，过拟合风险略增 | 更紧、更稳，容量不足时升 |
| `compress_dim` | 更丰富相似度（>1 时点积为多维） | 默认 1 省算力；先保持 |
| `kan_hidden_dims_*` | 更强非线性 | 数据少时宜小，或用 Simple |
| `num_knots` | 样条更灵活 | 过大易不稳 |
| `pooling=attention` | 自适应聚焦关键残基 | 默认 mean 更简单可复现 |

训练不稳时：先确认输入 \(d\) 与 ESM `hidden_dim` 一致（150M→640，否则 Linear/KAN 入口维错误）。

---

## 8. 实现边界（阅读代码时）

1. **无 batch 维**：`forward` 假定单对序列张量 `(L, d)`；batch 在 `DLPAffinity.forward` 里用 Python 循环 stack。
2. **未 mask padding**：若上游 pad，池化会吃进 pad 位；当前流水线多按真实长度切片（ESM encode 去掉 special token 后长度≈氨基酸数）。
3. `max_ag_len`：构造期字段，标准 `forward` 路径未强制截断。

---

## 9. 与文献叙事的关系

README 将 R2R 描述为「residue-level interaction modeling」。本仓库实现是**可学习压缩 + 交叉注意力型复现（非标准 Transformer multi-head）+ KAN 融合 + 池化**，不必与外部同名模块一一对应；以本文件与 `r2r.py` 为准。

---

## 参考实现

```47:66:release_package/models/r2r.py
    def forward(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        V_ab = self.kan_ab(X_ab)
        V_ag = self.kan_ag(X_ag)
        M_inter = torch.matmul(V_ab, V_ag.T)
        
        attn_weights = F.softmax(M_inter, dim=1)
        V_ag_proj = self.ag_proj(V_ag)
        M_proj = torch.matmul(attn_weights, V_ag_proj)
        
        M_concat = torch.cat([M_proj, V_ab], dim=1)
        H_AA = self.inter_kan(M_concat)
        # ... pooling ...
```
