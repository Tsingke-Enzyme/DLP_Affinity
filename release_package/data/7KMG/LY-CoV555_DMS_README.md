# LY-CoV555 DMS 数据预处理说明

## 数据来源
- **原始数据**: Bloom Lab GitHub - [SARS-CoV-2-RBD_MAP_LY-CoV555](https://github.com/jbloomlab/SARS-CoV-2-RBD_MAP_LY-CoV555)
- **论文**: Starr et al. (2021) Cell Reports Medicine. DOI: 10.1016/j.xcrm.2021.100255
- **PDB结构**: 7KMG

## 生成文件

### 模型输入格式 (推荐使用)
| 文件 | 描述 | 数据量 |
|------|------|--------|
| `LY-CoV555_DMS_train_model_input.csv` | 训练集，含抗体/抗原序列 | 1,556 |
| `LY-CoV555_DMS_val_model_input.csv` | 验证集，含抗体/抗原序列 | 389 |
| `LY-CoV555_DMS_wildtype_ref.csv` | 野生型参考 | 1 |

### 列说明
```
antibody_seq     : LY-CoV555 VH+VL序列 (226 aa, 固定)
antigen_seq      : RBD序列 (211 aa, 突变体)
mutation_id      : 突变标识 (如 E484K)
site             : RBD位点编号 (331-531)
wildtype         : 野生型氨基酸
mutation         : 突变后氨基酸
escape_fraction  : DMS escape值 [0-1]
```

### 辅助文件
| 文件 | 描述 |
|------|------|
| `LY-CoV555_antibody_seq.txt` | 抗体VH/VL序列 (FASTA格式) |
| `LY-CoV555_DMS_processed.csv` | 完整处理数据 |
| `LY-CoV555_train.csv` / `LY-CoV555_val.csv` | 详细版训练/验证集 |

## 数据统计

### Escape分布 (分层划分)
| 类别 | 训练集 | 验证集 |
|------|--------|--------|
| 低 (<0.01) | 1,479 | 369 |
| 中 (0.01-0.1) | 15 | 4 |
| 高 (≥0.1) | 62 | 16 |

### 关键逃逸位点
- **E484** (最强逃逸): E484K, E484P, E484Q 等
- **F490**: F490K, F490R, F490D
- **L452**: L452E, L452K

## 验证方法

由于escape fraction ≠ KD，建议的验证流程：

```python
# 1. 预测所有突变体的pKD
pred_mutant = model.predict(antibody_seq, mutant_rbd_seq)
pred_wt = model.predict(antibody_seq, wildtype_rbd_seq)

# 2. 计算预测的亲和力变化
delta_pred = pred_mutant - pred_wt  # 负值表示亲和力下降

# 3. 计算与escape的相关性
# escape高 → 亲和力低 → 期望负相关
from scipy.stats import spearmanr
r, p = spearmanr(delta_pred, escape_fraction)
# 期望 r < 0, p < 0.05
```

## 注意事项

1. **Escape ≠ KD**: Escape fraction测量的是FACS逃逸比例，不是热力学解离常数
2. **用途限制**: 此数据适合验证模型预测**突变效应方向**的能力，不适合作为绝对亲和力预测的benchmark
3. **论文表述建议**: "We validated the model's ability to predict mutational effects on binding using DMS data, achieving Spearman correlation of X.XX"

## 抗体序列

```
>LY-CoV555_VH (119 aa)
QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYNMDWVRQAPGQGLEWMGDINPNNGGTSYNQKFKGRVTVTVDKSTSTAYMELRSLRSDDTAVYYCARVRRSWYPFDYWGQGTLVTVSS

>LY-CoV555_VL (107 aa)
DIQMTQSPSSVSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQANSFPYTFGQGTKLEIK
```
