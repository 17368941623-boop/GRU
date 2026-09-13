# 0906：Thv最终模型架构统一训练包

本目录可以整体搬到Mac服务器。实验固定为预测未来第15步的Thv，历史窗口为60步，所有模型采用相同的数据、窗口、归一化、损失函数和验证规则。

## 一键运行

进入服务器上的`0906`目录，激活已经配置好的PyTorch环境后执行：

```bash
bash run_all_0906.sh
```

默认将任务分成两个互不重叠的模型家族，并同时运行5个训练进程：GRU家族3个、LSTM/对照家族2个。服务器资源允许时可调整为8个，但两个家族的总并发不应超过10个：

```bash
GRU_WORKERS=5 LSTM_WORKERS=3 bash run_all_0906.sh
```

若只希望完成训练和验证、暂时不读取0715-BACK测试集：

```bash
RUN_FULL_TEST=0 bash run_all_0906.sh
```

脚本支持断点恢复。重新执行时会核对每个模型的`metrics.json`与`best_model.pt`，完整结果自动跳过，不完整结果重新训练。不要在多个终端中重复启动同一个一键脚本。

若希望在两个终端分别管理进度，也可以执行下面两个训练脚本；它们写入同一个结果目录，但任务绝不重叠：

```bash
# 终端1：50次GRU家族训练
GRU_WORKERS=3 bash run_gru_family_0906.sh

# 终端2：35次LSTM/TCN家族训练
LSTM_WORKERS=2 bash run_lstm_family_0906.sh
```

两个脚本均结束后，再执行`bash run_all_0906.sh`即可自动复用全部已有模型，并生成验证汇总及完整测试结果。通常直接使用总入口最省事。

## 固定实验协议

- 预测目标：`Thv(t+15)-Thv(t)`，评价时恢复为`Thv(t+15)`。
- 采样周期：10 s；预测时域：150 s。
- lookback：60步。
- 公共预测起点历史要求：120步，与已完成的lookback对比保持相同窗口。
- 历史测量数据只截至当前时刻`t`。
- 未来输入只包括8个阀门的`u(t)`至`u(t+14)`，不包含未来温度、压力或流量。
- Standardizer仅使用训练数据拟合。
- 训练损失：快速降温样本权重为3的Huber损失，保留此前确定的动态关注目标。
- 早停、学习率调度和模型冻结：仅依据`Original 260501`完整验证集RMSE，不使用快降子集排名。
- 完成全部训练与验证汇总后，才读取`Original 0715-BACK`并进行冻结权重的完整序列推理。
- 0715-BACK结果只按完整测试RMSE排名，不使用快降子集。

## 训练模型

共85次训练：

| 模型 | 随机种子数 |
|---|---:|
| GRU baseline | 10 |
| LSTM baseline | 10 |
| TCN baseline | 5 |
| Parallel GRU–MLP–GNN | 10 |
| Parallel GRU–KAN–GNN | 10 |
| Parallel LSTM–MLP–GNN | 10 |
| Parallel LSTM–KAN–GNN | 10 |
| Serial MLP–GNN–GRU | 10 |
| Serial KAN–GNN–GRU | 10 |

10种子集合为`42,52,62,72,82,92,102,112,122,132`；TCN使用`42,62,82,102,122`。

两种并行LSTM模型与对应的并行GRU模型使用完全相同的GNN拓扑、边函数、未来阀门编码器和融合层，只替换历史编码器。因此可以分别回答两类问题：GNN相对各自循环基线是否有效，以及同一GNN条件下GRU还是LSTM更适合当前慢动态过程。

## 输出结果

所有结果写入`outputs_0906`：

- `development/`：模型权重、归一化参数、逐轮训练记录、完整验证预测和训练曲线。
- `validation_summary/validation_model_summary.csv`：完整验证集的模型汇总排名。
- `validation_summary/paired_vs_gru.csv`：与GRU同种子的配对比较。
- `validation_summary/paired_vs_lstm.csv`：与LSTM同种子的配对比较。
- `validation_summary/paired_structural_effects.csv`：串/并行、KAN/MLP以及GRU/LSTM骨干消融。
- `full_test_summary/test_model_summary.csv`：0715-BACK完整测试集最终汇总。
- `full_test_summary/test_seed_runs.csv`：每个模型和种子的完整测试指标。
- `full_test_summary/paired_test_vs_gru.csv`：完整测试集上与GRU的配对差异。
- `full_test_summary/paired_test_vs_lstm.csv`：完整测试集上与LSTM的配对差异。
- `full_test_summary/paired_test_structural_effects.csv`：完整测试集上的配对结构消融。
- 默认仅为seed=42保存各模型完整测试预测曲线CSV，便于后续绘制同一批次曲线图。可用`SAVE_PREDICTION_SEEDS=42,62`修改。

## 结果解释边界

本批次的模型权重完全不使用0715-BACK选择。但是0715-BACK此前已用于lookback比较，因此论文中应称其为事后测试/评估数据，而不再称为“从未查看的最终独立测试集”。若需要严格的最终独立测试结论，应另外保留一个未参与lookback与架构选择的新工况。

## 文件布局

- `model_code/`：模型、训练调度、验证汇总和完整测试代码。
- `run_all_0906.sh`：推荐的一键总入口；自动协调下面两个家族脚本并汇总结果。
- `run_gru_family_0906.sh`与`run_lstm_family_0906.sh`：可分别在两个终端运行的非重叠训练入口。
- `shared/`：无泄露窗口构造、归一化、训练循环和指标函数。
- `processed_data/`：本批次对应的训练、验证和测试数据。
- `requirements.txt`：Python依赖。
- `package_manifest.csv`：搬运前文件大小和SHA-256校验清单。
