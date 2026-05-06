# 彩蛋：SE 模块 Squeeze 操作改进实验

## 内容

- `train_se_flatten_comparison.py` — 实验代码：对比 Basic CNN、SE-CNN (GAP)、SE-CNN (Flatten) 及随机冻结对照组在 Fashion-MNIST 上的表现
- `results_se_flatten_comparison/` — 运行结果（训练曲线、准确率汇总表、Grad-CAM 可视化）
- `se_flatten_summary.pdf` — 实验总结文档

## 作用

本实验源于一个猜想：标准 SE 模块在浅层 CNN 特征上，Squeeze 阶段的全局平均池化（GAP）压缩过度，导致通道注意力机制未能充分激活。于是将 GAP 替换为 Flatten，保留完整空间信息送入 Excitation。第一轮实验发现 Flatten 版准确率明显提升，但参数量也暴增（3.2 万 → 44.2 万）。为排除"参数量堆砌"的嫌疑，第二轮引入随机冻结对照：SE 模块随机初始化后冻结不训练，可训练参数降回 3.2 万，结果准确率反而比 Baseline 还低——证明 Flatten SE 的提升来自真正学到的注意力，而非参数量的简单增长。

注：在总结文档中，我们对结论措辞进行了修正——不再断言 Flatten 的有效性一定来源于"通道-空间交互注意力"，而是将其作为推测性解释，并提及 CBAM 作为相关工作的参照，有待进一步对比实验验证。

猜想由我提出，代码编写和 PDF 撰写由 Claude Code 协助完成。
