# 阶段一：GRU 全局粗搜索

本训练包执行固定的 `lookback × hidden size` 验证集粗搜索：

- Lookback：15、30、45、60、75、90；
- GRU hidden size：32、64、128、256；
- Seed：42、52、62、72、82；
- 总计120次训练；
- 预测目标：`Thv(t+15)-Thv(t)`，评价时恢复为绝对 `Thv(t+15)`；
- 历史输入：`feature_catalog.json` 中全部54个原始信号；
- 未来输入：18个阀门的 `u(t)…u(t+14)`；
- 验证指标：完整 Original 验证过程的RMSE；
- 不使用PCMCI、构造特征或测试集。

## 服务器目录

将 `processed_data` 和整个 `1.0` 文件夹放在同一个服务器目录：

```text
工作目录/
├── processed_data/
└── 1.0/
    ├── code/
    ├── output/
    └── output_picture/
```

## 安装和运行

在包含 `processed_data` 与 `1.0` 的工作目录中执行：

```bash
python3 -m pip install -r "1.0/code/requirements.txt"
bash "1.0/code/run_stage1_global_search.sh"
```

脚本默认同时运行10个训练进程。若内存或MPS显存不足，可以改为5个：

```bash
WORKERS=5 bash "1.0/code/run_stage1_global_search.sh"
```

强制指定Apple MPS：

```bash
DEVICE=mps bash "1.0/code/run_stage1_global_search.sh"
```

训练中断后，直接再次运行同一命令会复用已经完成且通过审计的结果。只有确实需要覆盖全部结果时才使用：

```bash
OVERWRITE=1 bash "1.0/code/run_stage1_global_search.sh"
```

## 主要输出

服务器结果保存在 `1.0/output`：

- `development/`：每个参数和seed的模型、训练历史及验证预测；
- `validation_summary/validation_seed_runs.csv`：120次单独训练结果；
- `validation_summary/validation_cell_summary.csv`：每个热力图单元格的均值、SD与SE；
- `validation_summary/validation_rmse_mean_matrix.csv`：绘制热力图的RMSE均值矩阵；
- `validation_summary/coarse_candidate_cells.csv`：一标准误差范围内的候选单元格；
- `validation_summary/coarse_search_recommendation.json`：第二阶段精细搜索的候选边界和边界警告；
- `launcher_logs/`：环境、预检和10个工作进程日志。

训练、早停、汇总过程均不会读取 `test_clean.pkl` 或 `test_full_clean.pkl`。

