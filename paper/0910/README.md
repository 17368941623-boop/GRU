# 0910 模型架构 × 累计物理特征训练包

本目录可以整体搬到 Mac 服务器。主实验固定预测未来第 15 个采样点的 Thv，历史窗口为 60 步，并使用 `u(t)` 至 `u(t+14)` 的 8 个已知未来阀门指令。除阀门外，温度、压力、流量和构造特征均只使用到当前时刻 `t`。

## 一键运行

进入 `0910` 目录并激活已有的 PyTorch 环境，然后执行：

```bash
bash run_heatmap_training_0910.sh
```

默认同时运行 10 个训练进程，将 120 个固定实验均匀拆分为每个进程 12 次训练。5 个随机种子本身保持不变，并发数只影响调度速度，不改变实验内容。需要断点续跑时，直接再次执行同一命令；已通过完整性检查的结果会自动复用。

如需改变训练轮数等参数，可在命令前设置：

```bash
EPOCHS=120 PATIENCE=15 BATCH_SIZE=256 bash run_heatmap_training_0910.sh
```

不要同时启动两个总脚本并写入同一个 `outputs_0910`。只有明确需要覆盖已有结果时才使用：

```bash
OVERWRITE=1 bash run_heatmap_training_0910.sh
```

## 实验矩阵

模型为 3 种：

1. GRU baseline；
2. Parallel GRU-MLP-GNN；
3. Parallel GRU-KAN-GNN。

累计特征为 8 级：

1. 仅 20 个原始测量量；
2. 加入目标温度历史动态；
3. 再加入 Thv 模组耦合；
4. 再加入模组阀门动态；
5. 再加入主路状态差；
6. 再加入混气后制冷桥接；
7. 再加入混气热驱动代理；
8. 再加入 G 管主路流通代理，形成完整 20 个推荐构造特征。

所有单元格使用相同 5 个随机种子：`42, 52, 62, 72, 82`。总计 `3 × 8 × 5 = 120` 次训练。

累计顺序预先固定为“目标附近动态 → 模组耦合 → 上游管路信息”。因此热力图回答的是物理信息逐级引入后，各架构性能如何变化；它不能单独证明某一组特征具有与顺序无关的因果重要性。

## 公平性和无泄露约束

- 训练数据排除 `0617-ALL`。
- Standardizer、目标差值尺度和快速降温阈值只用训练数据拟合。
- 窗口不跨越原始文件边界。
- 所有配置使用相同公共预测起点（要求至少 120 步历史），防止不同输入长度造成样本不一致。
- 所有特征配置都保留相同的 42 维张量。未启用的构造特征在使用训练集统计量标准化后置零，因此同一架构内部参数量不变。
- 验证集仅为 Original 260501，完整验证集 RMSE 用于学习率调度、早停和检查点选择。
- 主训练和汇总程序不会反序列化 `test_clean.pkl` 或 `test_full_clean.pkl`，也不会读取 0715-BACK 指标。

## 主要输出

- `outputs_0910/development/`：每个单元格和种子的模型、Scaler、逐轮训练历史、验证预测及曲线。
- `outputs_0910/validation_summary/validation_seed_runs.csv`：120 次训练的逐种子原始指标。
- `validation_cell_summary.csv`：每个模型—特征单元格的均值、标准差和 95% t 区间。
- `paired_cumulative_feature_effects.csv`：同一模型中相邻累计阶段以及相对 raw20 的配对差值。
- `paired_architecture_effects.csv`：同一特征阶段中两种并行模型相对 GRU 基线的配对差值。
- `validation_rmse_mean_matrix.csv`、`validation_rmse_sd_matrix.csv`：可直接用于论文重绘的热力图源数据。
- `figures/`：绝对 RMSE 和相对 raw20 变化的 SVG、PDF、PNG、TIFF 候选图以及图形说明 JSON。

5 个种子适合控制计算量并观察方向一致性，但双侧 Wilcoxon 检验在该样本量下不能提供很强的显著性证据。论文中应优先报告均值、标准差、配对差值和 5 个种子的方向一致性，不应只写“显著提升”。

## 环境

若服务器环境尚未配置，可执行：

```bash
python3 -m pip install -r requirements.txt
```

Apple Silicon 会优先使用 MPS；不支持的算子允许回退到 CPU。代码同样兼容 CUDA 和纯 CPU。
