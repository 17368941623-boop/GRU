# Thv 残差（Delta）lookback 测试

本目录用于重新比较以下 7 个历史窗口：

`10 / 15 / 20 / 25 / 30 / 35 / 40`

预测任务为：

```text
Delta_Thv_h(t) = Thv(t+h) - Thv(t)
predicted_Thv(t+h) = Thv(t) + predicted_Delta_Thv_h(t)
```

采样周期为 10 s。命令行参数 `--predict-steps` 支持
`5 / 10 / 15 / 20 / 30`，分别对应 50 / 100 / 150 / 200 / 300 s。

## 已冻结的损失与模型选择规则

- 训练固定使用快速降温加权 Huber，不再做普通/加权损失消融。
- 快速降温阈值为训练集 `Delta_Thv_5` 的 10% 分位数，只从训练集拟合。
- 普通样本权重为 1，快速降温样本权重为 3。
- 每轮保存训练集与验证集的全区间 RMSE、快速降温 RMSE和加权 Huber。
- 早停和 lookback 排名以“快速降温验证 RMSE”为第一指标。
- 全验证集 RMSE作为整体准确性的护栏，同时保留并汇总。

## 输入结构

历史 LSTM 输入为截至当前时刻 `t` 的 20 个原始测量量，不加入构造特征。

另设一个未来控制分支，只输入以下 8 个阀门的 5 个控制动作：

```text
CV8312 / CV8311 / CV8310 / CV8313 / CV8300 / CV8351 / EC-V2 / COOLDOWN
```

时间对齐为：

```text
u(t), ..., u(t+h-1) -> Thv(t+h)
```

`u(t+h)` 与目标温度处在同一采样时刻，不用于解释此前 h 个区间的温度演化。

未来温度、压力、流量以及由它们构造的特征全部禁止进入输入。当前温度、压力和流量已经包含在历史 LSTM 的最后一个状态中，不在未来分支重复成“恒定的未来测量”。

这里得到的是“给定控制轨迹的条件预测模型”：离线验证使用记录到的阀门轨迹；以后进入 MPC 时，由优化器把候选阀门轨迹送入同一控制分支。因此该做法适合 MPC，但不能表述成不知道未来控制量的纯自主时序预测。

## 数据与泄露约束

- 训练集：`train_clean.pkl`，并在任何统计量拟合前排除 `0617-ALL`。
- 验证集：仅 Original `260501`，用于早停、checkpoint和lookback排名。
- 测试集：仅 Original `260118`；模型训练完成后评估固定 `source_row_index=6500..11500` 窗口。
- Scaler、Delta 标准化参数和快速降温阈值全部只由训练集拟合。
- 历史窗口和未来控制窗口都按 `file_id` 独立生成，不跨原始文件或过滤断点。
- 所有 lookback 使用至少具有 40 步历史且能提供完整 5 步控制轨迹的共同预测起点，保证比较样本完全一致。

每轮训练记录的是训练集和260501验证集RMSE。260118固定窗口只在最佳checkpoint已经由260501选定后计算，不参与梯度、早停或checkpoint选择。若再利用该测试窗口反向挑选lookback，它只能称为辅助评估集，不能继续声称是严格一次性测试集。

## Mac 环境

在 `0822` 目录安装依赖：

```bash
python -m pip install -r requirement.txt
```

代码会依次优先使用 CUDA、Apple MPS、CPU。Mac 主机检测到 MPS 时会自动使用。

## 训练命令

必须指定 lookback 和 seed。例如先完成 seed 42：

```bash
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 10 --seed 42
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 15 --seed 42
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 20 --seed 42
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 25 --seed 42
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 30 --seed 42
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 35 --seed 42
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 40 --seed 42
```

建议再分别使用 `--seed 62` 和 `--seed 82` 运行同样 7 个 lookback，共 21 次。不同 seed 或 lookback 的输出目录互不覆盖，可以在多个终端同时运行。

如果需要明确重跑已有实验：

```bash
python "残差lookback测试/train_thv_delta_lstm_lookback.py" --lookback 20 --seed 42 --overwrite
```

## 单次输出

输出目录：

```text
残差lookback测试/outputs/lookback_XX/seed_YY/
```

包含：

- `best_model.pt`：快速降温验证 RMSE 最佳轮次的模型、Scaler、阈值和输入列；
- `metrics.json`：全区间、快速降温、Persistence及方向准确率指标；
- `scalers.npz`：历史输入、未来阀门和 Delta 目标的标准化参数；
- `training_history.csv`：每轮训练/验证的全区间和快速降温 RMSE；
- `validation_predictions.csv`：恢复后的温度预测、Delta 预测、快速降温标记和每个未来控制动作；
- `training_curve.png`；
- `validation_prediction.png`；
- `validation_delta.png`。

训练结束时命令行会输出：

```text
FINAL_TRAIN_RMSE_K=...
FINAL_TRAIN_RAPID_RMSE_K=...
FINAL_VALIDATION_RMSE_K=...
FINAL_RAPID_VALIDATION_RMSE_K=...
MODEL_SAVED=...
```

## 汇总

完成实验后运行：

```bash
python "残差lookback测试/summarize_delta_lookback.py"
```

生成：

- `lookback_all_runs.csv`；
- `lookback_summary.csv`；
- `lookback_rmse_comparison.png`；
- 命令行输出快速降温验证 RMSE 最优的 lookback。

## LSTM 与 GRU 基础循环层对比

在进行完整 lookback 扫描或人工特征消融前，可先判断 LSTM 和 GRU 哪一个更适合作为历史观测编码器。对比脚本保持以下内容完全相同：

- 训练集、验证集和共同预测起点；
- Delta 目标与恢复方式；
- 快速降温加权 Huber 和训练集阈值；
- `u(t)...u(t+4)` 未来阀门控制分支；
- hidden size、融合层、优化器、学习率、batch size 和早停规则。

唯一变化是历史循环单元为 LSTM 或 GRU。首轮建议只测试原最佳窗口 20 和更长窗口 40，以检验长历史增加后两种结构的稳定性，共 12 次：

```bash
python "残差lookback测试/train_thv_delta_rnn_compare.py" --model lstm --lookback 20 --seed 42
python "残差lookback测试/train_thv_delta_rnn_compare.py" --model gru  --lookback 20 --seed 42
python "残差lookback测试/train_thv_delta_rnn_compare.py" --model lstm --lookback 40 --seed 42
python "残差lookback测试/train_thv_delta_rnn_compare.py" --model gru  --lookback 40 --seed 42
```

将上述命令的 seed 分别替换为 `62` 和 `82`。不同模型输出到：

```text
残差lookback测试/rnn_compare_outputs/lstm/lookback_XX/seed_YY/
残差lookback测试/rnn_compare_outputs/gru/lookback_XX/seed_YY/
```

每次实验除预测指标外，还记录可训练参数量、训练总时间和平均每轮耗时。这里固定相同 hidden size，因此 LSTM 参数量通常高于 GRU；结果解释时需要同时考虑快速降温 RMSE、全区间 RMSE、种子稳定性和效率。

完成后运行：

```bash
python "残差lookback测试/summarize_rnn_compare.py"
```

汇总会生成：

- `architecture_all_runs.csv`；
- `architecture_by_lookback.csv`；
- `architecture_score.csv`；
- `architecture_paired_differences.csv`；
- `long_history_gain.csv`：分别统计 LSTM/GRU 从 lookback 20 增加到 40 后的 RMSE 变化；
- `lstm_gru_comparison.png`。

主判据仍为快速降温验证 RMSE。若两者差异小于种子波动，优先选择参数更少、训练更快的 GRU；若 LSTM 在 lookback 40 下仍稳定取得明确优势，再确定 LSTM 更适合长历史趋势。

## 已完成的循环结构结论（2026-08-29）

以下LSTM/GRU数字来自旧划分（验证=260118、测试=260501），只保留为历史结构筛选记录；交换划分后的正式lookback结果必须重新训练，不能与这些数值直接混合。

在 lookback `20 / 40`、seed `42 / 62 / 82` 的 12 次严格配对结果中：

- GRU 快速降温验证 RMSE：`0.16386 +/- 0.03031 K`；
- LSTM 快速降温验证 RMSE：`0.18739 +/- 0.04500 K`；
- GRU 全验证集 RMSE：`0.06369 +/- 0.00854 K`；
- LSTM 全验证集 RMSE：`0.07356 +/- 0.01446 K`；
- GRU 使用 `24097` 个参数，LSTM 使用 `29601` 个参数。

因此后续 lookback 选择固定采用 GRU。GRU 的 lookback=40 明显优于 lookback=20，但 40 是原扫描上界，不能据此直接宣称 40 为最优窗口。

## GRU 正式 lookback 扫描

正式扫描范围扩展为：

```text
10 / 15 / 20 / 25 / 30 / 35 / 40 / 45 / 50 / 55 / 60
```

所有 lookback 都只使用具备至少 60 步历史的共同预测起点，因此样本数、预测时刻和标签严格一致。每个变量独立使用仅由训练集拟合的 Standardizer；温度、压力、流量和阀门绝不共用一个尺度。

正式 lookback 选择的第一指标不是全曲线 RMSE，而是 Original `260501` 验证曲线中满足以下条件的快速降温窗口：

```text
Delta_Thv_5 <= 训练集 Delta_Thv_5 的 10% 分位数阈值
```

快速窗口的实际数量由程序逐次输出并保存在指标文件中。全验证曲线 RMSE只作为第二指标和准确性护栏。

数据集术语必须保持一致：lookback早停和正式排名使用 Original `260501` 验证集。Original `260118` 是测试集，训练结束后从 `test_full_clean.pkl` 按原始 `source_row_index=6500..11500` 提取固定快降窗口，报告该窗口全体预测点RMSE；同时再报告其中满足训练集快速阈值的子集RMSE。测试阈值绝不从260118重新拟合。

先运行 seed 42。以下命令默认 `--predict-steps 5`；例如预测未来15步时，
给每条命令增加 `--predict-steps 15`：

```bash
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 10 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 15 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 20 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 25 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 30 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 35 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 40 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 45 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 50 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 55 --seed 42
python "残差lookback测试/train_thv_delta_gru_lookback.py" --lookback 60 --seed 42
```

随后将 seed 替换为 `62` 和 `82`，共完成 33 次。不同任务可并行运行，但不要对同一个 `(lookback, seed)` 同时启动两次。

单次结果保存在：

```text
残差lookback测试/gru_lookback_outputs_val260501_test260118/horizon_HH/gru/lookback_XX/seed_YY/
```

除模型、预测图和逐点预测外，新增保留：

- `training_history.csv`：每一轮训练集/验证集的全区间与快速降温 RMSE、加权 Huber 和学习率；
- `rapid_validation_predictions.csv`：仅保留由训练集阈值定义的快速降温验证窗口，用于主RMSE和论文局部曲线；
- `normalization_report.csv`：每个输入变量的训练/验证范围、训练集均值与标准差、缩放后范围，明确 Scaler 只由训练集拟合；
- `leakage_audit.json`：记录预测量、标签对齐、输入时间范围、未来控制条件和数据划分；
- `test_260118_hHH_window_6500_11500_predictions.csv`：固定测试窗口逐点预测；
- `test_260118_hHH_window_6500_11500_metrics.json`：固定窗口RMSE及其中训练阈值快速子集RMSE；
- `test_260118_hHH_window_6500_11500_prediction.png` 和 `..._delta.png`：测试窗口曲线。
- `validation_segment_metrics.csv`：每个连续 `file_id` 的 RMSE、MAE、P95 和最大误差；
- `metrics.json`：最佳 checkpoint、最后完成轮次、Persistence、方向准确率、耗时和参数量。

完成后汇总：

```bash
python "残差lookback测试/summarize_gru_lookback.py" --predict-steps 5
```

汇总结果包括全部逐轮曲线 `gru_epoch_history_all_runs.csv`、正式论文表 `paper_evidence_lookback_table.csv`、各 lookback 均值/标准差、相对最优窗口的同 seed 配对差值以及比较图。只有 11 个 lookback 的 `42 / 62 / 82` 全部完成时，命令行才会输出 `FORMAL_SELECTION_READY=true`。
