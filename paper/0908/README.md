# 0908：Thv 多步预测的物理—时延并行 GNN + KAN + GRU

本目录是一套可直接搬到服务器的训练包。它沿用当前冻结的数据划分和 `lookback=60`，默认直接输出
未来 15 步（150 s）的 `Thv` 曲线；把 `PredictSteps` 改成 30，即输出未来 5 min（30×10 s）的曲线。

## 已固定的输入与输出

- 历史输入：`t-59 ... t` 的 20 个原始变量 + 20 个已验证的构造特征。
- 图输入节点：20 个原始变量；边为 `source(t-τ) → destination(t)`。
- 未来输入：仅 8 个可规划阀门 `u(t) ... u(t+H-1)`。
- 输出：一次性直接输出 `Thv(t+1) ... Thv(t+H)` 相对 `Thv(t)` 的残差，再还原为绝对温度。
- 第 `k` 个输出只能读取 `u(t) ... u(t+k-1)`；不读取未来温度、压力、流量，也不递归输入预测温度。

这种定义适合后续 MPC：优化器给出候选未来阀门序列，模型返回对应的整段温度轨迹。历史中没有
“未来阀门”并不会造成预测延迟；在线时未来阀门来自 MPC 的候选控制方案，而不是传感器测量。

## 静态物理关系

图的人工先验在 `graphs/static_physical_graph.json`，解释见 `docs/static_physical_relations.md`。它包含：

- G1/G2 入口阀 → `TE8310/PT8310`；
- `CV8310` 与 4.6 K 的 `CV8351` → `FT8351/TE8351/PT8351`；
- `TE8351/PT8351/FT8351` → `TE8352/PT8352` → `TE8353`；
- `TE8353/FT8351/EC-V2/COOLDOWN` → `Thv`；
- 关键状态的自滞后边。

`CV8311/CV8312` 只作为冷屏分流对主路流量的竞争关系，置信度设为中等。`Tef/Tcd/DTbr` 的位置
说明不足，所以保留为 GRU 输入，不强行加入静态边。

## 服务器一键启动（推荐）

把整个 `0908/` 目录复制到 Linux 服务器；不要只复制代码，因为训练依赖其中已经冻结的
`processed_data/`、`graphs/` 和实验协议。服务器的 `pytorch` conda 环境需提前安装与服务器
CUDA 匹配的 PyTorch。进入目录后运行：

```bash
chmod +x *.sh
bash start_server_training.sh
```

脚本会在后台依次完成：环境依赖检查/安装（不会替换 PyTorch）、预检、因果建图、80 次开发集训练、
验证集汇总。默认最多启用 5 个训练进程，并按 `cuda:0`、`cuda:1`……每张卡一个进程；若只有一张卡，
只启动一个进程，不会让 5 个进程挤在同一张卡上。查看进度：

```bash
tail -f logs/server_pipeline.console.log
cat logs/server_pipeline.status
```

中止任务可运行 `bash stop_server_training.sh`。再次启动会跳过已有完整 checkpoint，重做被中断的
单次实验，因此可以断点续跑。服务器环境名、预测长度等可在启动命令前覆盖，例如：

```bash
CONDA_ENV=pytorch PREDICT_STEPS=30 BOOTSTRAP=50 bash start_server_training.sh
```

默认 `PREDICT_STEPS=15`、`BOOTSTRAP=20`。自动并发还会按每个进程预留约 3 GB 可用内存，
可用 `SERVER_WORKERS` 显式覆盖。若要重建已经完整生成的因果图，增加
`FORCE_CAUSAL=1`。如果服务器环境已装好全部依赖，可用 `INSTALL_DEPS=0` 跳过安装。

一键脚本**不会读取冻结测试集**。待 80 次开发实验全部完成、人工查看验证汇总并锁定方案后，才运行：

```bash
PYTHON_BIN=python PREDICT_STEPS=15 DEVICE=cuda:0 bash run_frozen_test.sh
```

## 分阶段运行

先安装依赖，然后按以下顺序执行。PowerShell 示例：

```powershell
pip install -r requirements.txt
.\run_preflight.ps1 -PythonPath python
.\run_causal_discovery.ps1 -PythonPath python -Bootstrap 20
.\run_development_5workers.ps1 -PythonPath python -PredictSteps 15 -Device cuda
.\run_frozen_test.ps1 -PythonPath python -PredictSteps 15 -Device cuda
```

未来 5 min 版本只需把最后两条命令的 `PredictSteps` 改为 30。建议先跑 H=15，保持与 0907 结果
可比；确认方案后再跑 H=30。正式论文可把 bootstrap 增加到 50。

Linux 服务器也可使用同名 `.sh` 脚本逐阶段执行，例如：

```bash
PYTHON_BIN=python BOOTSTRAP=20 bash run_causal_discovery.sh
PYTHON_BIN=python PREDICT_STEPS=15 DEVICE=cuda bash run_development_5workers.sh
PYTHON_BIN=python PREDICT_STEPS=15 DEVICE=cuda:0 bash run_frozen_test.sh
```

本地主机低内存条件下使用 `run_local_background.ps1`。该脚本只启用一个 CUDA 训练进程，依次完成
预检、PCMCI、80 个开发集训练和验证汇总；状态保存在 `logs/local_pipeline_status.json`。它不会自动
打开冻结测试集，待开发训练完成并确认验证汇总后再运行 `run_frozen_test.ps1`。

## 固定实验矩阵

每个方案使用 10 个相同随机种子，共 80 次训练；并行进程数由可见 GPU 数量决定，上限默认是 5：

1. `gru_full20`：无图基线；
2. `physical_all_lags_kan`：物理边的允许滞后区间全部展开；
3. `cd_select_lag_kan`：仅 PCMCI 稳定边；
4. `dkcdv_select_lag_kan`：主模型，物理验证的 PCMCI 图；
5. `dkcdl_select_lag_kan`：仅保留物理允许变量对；
6. `dkcdv_select_lag_mlp`：KAN/MLP 对照；
7. `dkcdv_shuffled_lag_kan`：相同变量对、打乱滞后的负对照；
8. `random_same_size_kan`：相同边数随机图负对照。

因果发现只读取未扩增的训练运行，并排除 0617；训练父运行被固定分成互不重叠的相关性筛选组和
PCMCI 推断组。先在候选边上运行一次受限 PC1/MCI，再固定完整拟合得到的条件父集，在原始连续段
层面做 MCI bootstrap；不会在每个 bootstrap 中重新搜索全部 12,000 条变量—滞后组合。验证集和测试集
不参与建图。训练只读取
`train_clean.pkl` 和 `val_clean.pkl`。80 个验证实验完整后，`summarize.py` 才生成冻结清单；
`evaluate_test.py` 默认要求该清单存在，之后才读取 `test_full_clean.pkl`。

主模型训练完成后，可导出阀门 KAN 曲线：

```bash
python model_code/export_kan_curves.py --checkpoint <某个best_model.pt>
```

它分别保存未来控制分支和 GNN 阀门边的“开度—导通指标”CSV。该曲线是局部可解释关系，
不能解释成整个预测网络对温度的全局单调关系。

## 论文需保存的结果

每次训练保存模型权重、训练曲线数据、归一化参数、逐点验证预测和完整配置。汇总至少报告：

- 整段 trajectory RMSE/MAE；
- 每个预测时刻的 RMSE/MAE 曲线及最终时刻 RMSE；
- 阀门动作窗口和快降窗口的 RMSE；
- 最强降温幅值误差、发生时间误差、曲线积分误差；
- 10 种子均值±标准差、配对 Wilcoxon 检验及 Holm 校正；
- PCMCI 的 effect、p/q 值、bootstrap 稳定性，以及最终边数。

主图至少需要：真实/预测 5 min 轨迹示例、RMSE 随预测时间曲线、8 方案多种子分布、最终
`变量—滞后`图、KAN 阀门导通曲线。不能只给一个总 RMSE，也不能用测试集重新选择边或模型。

## 目录

- `processed_data/`：已经复制的冻结数据；
- `graphs/static_physical_graph.json`：人工物理先验；
- `graphs/generated/`：因果发现和全部图对照；
- `model_code/`：建图、训练、汇总和测试代码；
- `outputs/development/`：训练与验证输出；
- `outputs/final_test/`：冻结模型测试输出；
- `reports/preflight_report.json`：数据、形状、单调性和未来控制泄漏检查。
