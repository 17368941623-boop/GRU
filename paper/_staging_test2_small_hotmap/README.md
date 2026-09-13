# 阶段二：GRU 局部精细热力图搜索

本训练包在固定的困难降温验证片段 H1–H4 上，对 `lookback × hidden size`
进行局部精细搜索：

- Lookback：50、60、70、80、90、100；
- GRU hidden size：128、192、256、320；
- Seed：42、52、62、72、82；
- 24个参数单元格，每个5个种子，共120次训练；
- 默认同时启动24个训练进程，每个进程顺序完成5次训练；
- 预测目标：`Thv(t+15)-Thv(t)`，评价时恢复为绝对 `Thv(t+15)`；
- 历史输入：`feature_catalog.json` 中启用的全部原始信号；
- 未来输入：全部启用阀门的 `u(t)…u(t+14)`；
- checkpoint早停指标：完整 Original 260428 验证过程RMSE；
- 热力图主指标：H1–H4共1440个目标点的汇总RMSE；
- 不使用PCMCI、构造特征或测试集。

## 固定困难降温片段

下列时间均为未来15步预测的目标时刻，起止时间均包含：

| 片段 | 起始时间 | 结束时间 | 目标点数 |
|---|---|---|---:|
| H1 | 2026-04-29 04:37:26 | 2026-04-29 05:37:16 | 360 |
| H2 | 2026-04-29 08:40:16 | 2026-04-29 09:40:06 | 360 |
| H3 | 2026-04-29 10:24:26 | 2026-04-29 11:24:16 | 360 |
| H4 | 2026-04-29 14:18:46 | 2026-04-29 15:18:36 | 360 |

每次训练保存完整验证预测，并单独计算H1、H2、H3、H4及四段汇总指标。
程序会严格检查每段必须包含360个有效预测目标，且四段不得重叠。

## 服务器目录

将 `processed_data` 和整个 `test2 small hotmap` 文件夹放在同一个服务器目录：

```text
工作目录/
├── processed_data/
└── test2 small hotmap/
    ├── code/
    ├── output/
    └── output_picture/
```

## 安装和运行

在包含上述两个文件夹的工作目录执行：

```bash
python3 -m pip install -r "test2 small hotmap/code/requirements.txt"
bash "test2 small hotmap/code/run_stage2_fine_search.sh"
```

脚本默认并发24个训练。需要降低并发时，例如改为10个：

```bash
WORKERS=10 bash "test2 small hotmap/code/run_stage2_fine_search.sh"
```

强制使用Apple MPS：

```bash
DEVICE=mps bash "test2 small hotmap/code/run_stage2_fine_search.sh"
```

训练中断后可直接重新执行同一命令，已完成且通过审计的模型会被复用。
只有确实需要覆盖全部结果时才运行：

```bash
OVERWRITE=1 bash "test2 small hotmap/code/run_stage2_fine_search.sh"
```

## 主要输出

结果保存在 `test2 small hotmap/output`：

- `development/`：120个模型、训练历史、scaler和完整验证预测；
- `validation_summary/validation_seed_runs.csv`：每次训练的完整验证及H1–H4指标；
- `validation_summary/validation_cell_summary.csv`：24个单元格的均值、SD、SE和95% CI；
- `validation_summary/hard_cooling_rmse_mean_matrix.csv`：论文主热力图数据；
- `validation_summary/hard_cooling_rmse_sd_matrix.csv`：困难片段种子波动；
- `validation_summary/full_validation_rmse_mean_matrix.csv`：完整验证集辅助热力图；
- `validation_summary/fine_candidate_cells.csv`：困难片段最优值一个SE内的候选；
- `validation_summary/fine_search_recommendation.json`：最终候选及边界检查；
- `launcher_logs/`：环境、预检、24个工作进程和汇总日志。

训练、早停和汇总过程均不会反序列化 `test_clean.pkl` 或
`test_full_clean.pkl`。五个种子只描述训练随机性，不代表五次独立工况。
