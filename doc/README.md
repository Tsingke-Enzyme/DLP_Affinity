# DLP-Affinity 文档索引

本目录面向工程与算法读者，说明亲和力微调流水线与任务头方法。

| 文档 | 内容 |
|------|------|
| [训练指南](./training-guide.md) | 问题定义、模型总览、训练过程与原理、可调参数、与当前 Argo 任务对应关系 |
| [R2R 方法详解](./r2r.md) | 残基对残基交互（KAN 压缩、attention 聚合、池化） |
| [GSPE 方法详解](./gspe.md) | 全局随机投影嵌入（投影–排序–距离） |

代码入口：`release_package/`（`train.py`、`models/`、`configs/config.py`）。
