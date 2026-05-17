# Deep Learning for Protein Complex Prediction and Design

> 📖 论文笔记仓库 | PhD Thesis Notes

**原论文**：*Deep Learning for Protein Complex Prediction and Design*
**作者**：Ziwei Xie（谢子威）
**机构**：Toyota Technological Institute at Chicago（TTIC）
**导师**：Jinbo Xu（徐进波）
**时间**：2026年3月
**arXiv**：[2605.11189](https://arxiv.org/abs/2605.11189)

---

## 论文简介

本论文围绕**蛋白质复合物结构预测与设计**这一计算结构生物学核心问题，提出了三个基于深度学习的方法，覆盖从界面接触预测、复合物结构预测到固定骨架序列设计的完整链条。

```
蛋白质复合物建模
├── 正问题（结构预测）
│   ├── GLINTER  →  界面接触预测（辅助对接）
│   └── ESMPair  →  同源蛋白配对（改进复合物结构预测）
└── 逆问题（序列设计）
    └── RedNet   →  选择性蛋白质结合子设计
```

---

## 目录

| 文件 | 内容 |
|------|------|
| [00_overview.md](./00_overview.md) | 研究背景、动机与三大贡献总览 |
| [01_GLINTER.md](./01_GLINTER.md) | 第2章：图学习蛋白质界面接触预测 |
| [02_ESMPair.md](./02_ESMPair.md) | 第3章：蛋白质语言模型辅助异源二聚体结构预测 |
| [03_RedNet.md](./03_RedNet.md) | 第4章：对比解码设计选择性蛋白质结合子 |
| [04_conclusions.md](./04_conclusions.md) | 第5章：总结与未来方向 |

---

## 核心贡献一览

| 方法 | 任务 | 核心创新 | 发表 |
|------|------|---------|------|
| **GLINTER** | 界面接触预测 | 多尺度结构图 + MSA Transformer共进化信号融合 | *Bioinformatics* 2022 |
| **ESMPair** | 异源二聚体结构预测 | PLM column attention评分驱动的同源蛋白配对 | *Briefings in Bioinformatics* 2023 |
| **RedNet** | 固定骨架序列设计 | 多尺度图Transformer + 对比解码优化选择性 | 论文新工作 |

---

## 关键结果速览

- **GLINTER**：在CASP-CAPRI基准上，Top-10精度达54%（同源二聚体）/ 44%（异源二聚体），大幅超越ComplexContact、DeepHomo等方法
- **ESMPair**：pConf70测试集DockQ 0.259，成功率42.4%；三策略集成后DockQ 0.285，成功率46.8%
- **RedNet**：异源二聚体Native Sequence Recovery 0.43（vs ProteinMPNN 0.37），对比解码显著提升结合选择性
