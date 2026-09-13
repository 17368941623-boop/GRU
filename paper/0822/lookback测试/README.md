# Thv LSTM lookback测试

本目录用于比较12种历史窗口长度：

`5 / 10 / 15 / 20 / 25 / 30 / 35 / 40 / 45 / 50 / 55 / 60`

预测任务为根据截至当前时刻的原始测量序列，直接预测 `Thv(t+5)`，即50秒后的Thv。

## 数据和泄露约束

- 训练：排除 `0617-ALL` 后的训练数据。
- 验证：仅 Original `260118`。
- 测试：`260501` 在lookback选择阶段完全不读取。
- Scaler只用排除0617后的训练集拟合。
- 所有窗口均按 `file_id` 独立生成，不能跨原始文件或过滤断点。
- 所有lookback使用至少具有60步历史的共同预测起点，确保样本和标签完全一致。
- 输入只使用20个原始测量量，不使用任何构造特征或未来标签。

## 环境

在0822目录安装依赖：

```bash
python -m pip install -r requirement.txt
```

Mac主机将自动优先使用PyTorch MPS；如果MPS不可用则使用CPU。

## 训练命令

必须通过参数指定lookback和seed：

```bash
python "lookback测试/train_thv_lstm_lookback.py" --lookback 5 --seed 42
python "lookback测试/train_thv_lstm_lookback.py" --lookback 30 --seed 62
python "lookback测试/train_thv_lstm_lookback.py" --lookback 60 --seed 82
```

建议每个lookback运行 `42 / 62 / 82` 三个种子，共36次训练。

如需调整训练轮数等固定参数：

```bash
python "lookback测试/train_thv_lstm_lookback.py" \
  --lookback 20 \
  --seed 42 \
  --epochs 100 \
  --batch-size 256 \
  --patience 12
```

同一lookback和seed已经存在时，程序默认拒绝覆盖；确认重跑时添加：

```bash
python "lookback测试/train_thv_lstm_lookback.py" --lookback 20 --seed 42 --overwrite
```

## 单次输出

输出位置为：

```text
lookback测试/outputs/lookback_XX/seed_YY/
```

包含：

- `best_model.pt`：最佳验证RMSE对应的LSTM模型和Scaler参数；
- `metrics.json`：完整验证、动态验证及Persistence指标；
- `scalers.npz`：输入和标签标准化参数；
- `training_history.csv`；
- `validation_predictions.csv`；
- `training_curve.png`；
- `validation_prediction.png`。

训练结束时命令行会明确输出：

```text
FINAL_VALIDATION_RMSE_K=...
MODEL_SAVED=...
```

## 汇总

完成多个lookback/seed后运行：

```bash
python "lookback测试/summarize_lookback.py"
```

生成：

- `lookback_all_runs.csv`；
- `lookback_summary.csv`；
- `lookback_rmse_comparison.png`；
- 命令行输出平均验证RMSE最小的lookback。
