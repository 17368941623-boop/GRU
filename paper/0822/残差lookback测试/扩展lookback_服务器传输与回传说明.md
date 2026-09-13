# GRU 扩展 lookback 测试（预测未来 15 步）

## 为什么需要扩展

原扫描范围为 10–60 步，三种随机种子的快速降温验证集平均 RMSE 在
lookback=60 时最低。由于最优值落在搜索上边界，论文中不宜直接把 60
称为最优历史长度，因此增加一次边界确认实验。

扩展候选为 `60 / 75 / 90 / 105 / 120`。五个候选统一使用
`common_origin_lookback=120`，从而严格保证验证标签、预测时刻和样本数一致。
旧扫描中的 lookback=60 使用的是 60 步公共起点，不能代替本次重新训练的
lookback=60。

## 搬到 Mac 服务器的代码

从本文件所在文件夹搬运以下五个文件：

1. `run_gru_extended_lookback_h15.sh`
2. `train_thv_delta_gru_lookback_extended.py`
3. `train_thv_delta_rnn_compare.py`
4. `train_thv_delta_lstm_lookback.py`
5. `summarize_gru_lookback.py`

服务器上还应保留当前版本的 `0822/processed_data/`。如果它已存在且没有被
重新生成，则不必再次上传。

## 服务器运行命令

在 `0822` 目录执行：

```bash
bash "残差lookback测试/run_gru_extended_lookback_h15.sh"
```

脚本会顺序完成 15 次训练，并在最后自动汇总。中断后重新执行时，已经存在
`metrics.json` 的完整实验会自动跳过。正常汇总应输出
`FORMAL_SELECTION_READY=true` 和 `MISSING_RUN_COUNT=0`。

## 从服务器回传

先完整回传以下六个汇总文件：

- `gru_lookback_all_runs.csv`
- `gru_lookback_summary.csv`
- `gru_lookback_paired_vs_best.csv`
- `gru_epoch_history_all_runs.csv`
- `paper_evidence_lookback_table.csv`
- `gru_lookback_rmse_comparison.png`

它们位于：

```text
残差lookback测试/gru_lookback_extended_val260501_test0715BACK/horizon_15/
```

同时回传五个 lookback、三个种子共 15 个运行目录中的：

- `metrics.json`
- `training_history.csv`
- `normalization_report.csv`
- `leakage_audit.json`
- 0715-BACK 固定窗口的 `metrics.json`

汇总确定最佳 lookback 后，再回传该 lookback 三个种子的完整运行目录，包括
`best_model.pt`、`scalers.npz`、预测图和逐点预测。体积较大的完整
`validation_predictions.csv` 可以不回传；请保留
`rapid_validation_predictions.csv` 和 0715-BACK 固定窗口预测文件。

## 选择与停止规则

只使用 Original 260501 快速降温验证 RMSE 的三种子均值选择 lookback，完整
验证 RMSE 作为准确性护栏；0715-BACK 只作最终诊断，不参与排序。若最佳点不再
位于 120，或 120 相比 105 的改善小于随机种子波动，可停止扩展。若 120 仍有
稳定、明显优势，再决定是否做最后一轮更长历史测试。
