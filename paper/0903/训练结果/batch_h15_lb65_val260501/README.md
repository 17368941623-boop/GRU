# Thv h15 / lookback 65 同协议模型批次

本文件夹汇总预测 `Thv(t+15)` 的同协议验证结果。历史窗口为 65，所有模型使用相同的训练/验证划分、未来阀门输入对齐和训练集拟合的归一化器。

## 数据与指标口径

- 训练数据：`train_clean.pkl`，按既有加载规则排除 `0617-ALL`。
- 验证数据：仅 `Original 260501`。
- 独立测试数据：`Original 0715-BACK`，本批次未读取。
- 预测目标：预测 15 步温度增量，并恢复为绝对温度计算指标。
- 已知未来控制：`u(t)` 至 `u(t+14)`；不输入任何未来温度、压力或流量。
- 快降样本：`Thv(t+15)-Thv(t) <= -0.585602 K`。阈值只由训练集的 10% 分位数确定。
- 每次验证共 25,600 个窗口，其中快降窗口 1,596 个，占 6.234%。

## 文件结构

- `raw_runs/recurrent_baselines_10seeds`：GRU 与 LSTM，各 10 个种子；从 0822 的已完成结果复制，原文件保留。
- `raw_runs/tcn_baseline_5seeds`：TCN，5 个种子。
- `raw_runs/graph_models_10seeds`：四种 GNN 组合，各 10 个种子。
- `summary/ablation_seed_runs.csv`：65 次训练的逐种子统一指标。
- `summary/ablation_model_summary.csv`：各模型均值、标准差及 95% Student-t 置信区间。
- `summary/paired_vs_gru.csv`：各模型与 GRU 的同种子配对比较。
- `summary/paired_structural_effects.csv`：串/并行及 KAN/MLP 的配对消融。
- `summary/batch_audit.json`：完整性、数据划分和无测试集读取审计。
- `消融实验报告.md`：可用于论文整理的中文结论与证据边界。
- `summarize_ablation.py`：重新生成上述统计表的脚本，仅读取验证结果。

## 未并入本批次的结果

旧的 lookback=60、三种子 `0715-BACK` 最终测试结果使用了不同 lookback、不同超参数和不同数据划分，因此保留在原位置，不与本批次的验证结果合并排名。0822 中被中止批次留下的单个 TCN 种子也未纳入；本批次采用 0903 新完成的五种子 TCN 结果。

