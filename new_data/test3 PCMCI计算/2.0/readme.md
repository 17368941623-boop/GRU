# 图约束与模型消融 2.0

训练时，将 `C:\Users\Administrator\Desktop\new_data` 下的 `processed_data` 和当前 `2.0` 文件夹一起搬到服务器上的同一工作目录。

目录用途：

- `code/`：图约束配置、PyTorch 读取接口、检查脚本，后续也在这里放并行 GNN–KAN–GRU 训练代码。
- `output/`：服务器训练结果搬回本地后的存放位置。
- `output_picture/`：论文图片及其生成代码。

## 本版已经冻结的内容

1. 静态物理拓扑以已确认的 `static_physical_graph_v2_review` 为准。
2. 逻辑图保留虚拟节点 `H_mix`；实现时折叠为一条 `A管 -> Thv` 的条件 KAN 边。`EC-V2、COOLDOWN、FC-V1` 是该边函数的条件输入，不作为三条普通直连边。
3. `A管 -> Thv` 使用当前 A 管温度水平，以及 13–17 步（130–170 s）的温度变化 lag 池。
4. `COOLDOWN` 使用当前开度水平，以及 9–13 步（90–130 s）的开度变化 lag 池。
5. `EC-V2` 和 `FC-V1` 不写死单一时延。保留当前开度，由控制历史编码器读取 `t-59..t` 的 60 个动作变化并学习摘要。
6. `TE8310 -> TE8351`、`PT8310 -> PT8351` 仍是静态物理边；三支路 KAN 只生成这两条边的动态门控 `alpha_T`、`alpha_P`，不会产生虚假的支路物料汇合边。
7. PCMCI 的 lag 是允许读取的候选池，不是强制的唯一因果时延。池内权重由训练集端到端学习。
8. 非静态物理图中的 PCMCI 边单独保存，默认关闭，只在 `A7` 消融中通过可学习门控启用。

机器可读文件位于 `code/config/`：

- `graph_constraints_v2.json`：节点、`edge_index`、关系类型、分表示 lag 池、两组 KAN 特殊边函数及可选 PCMCI 边。
- `ablation_matrix_v2.json`：A0–A7 公平消融方案、阶段化种子计划和必须保存的指标。
- `graph_constraint_audit_v2.json`：配置生成审计和计数。
- `source_snapshot_v2.json`：静态关系与本次 PCMCI 稳定结果快照，保证以后能追溯 2.0 配置来源。

## 检查

Windows：

```powershell
cd "test3 PCMCI计算\2.0"
.\code\run_graph_checks.ps1
```

Linux 服务器：

```bash
cd "test3 PCMCI计算/2.0"
bash code/run_graph_checks.sh
```

默认 Conda 环境名为 `pytorch`；环境名不同时设置 `CONDA_ENV_NAME`。

## 训练代码的读取方式

```python
from graph_constraints import load_config, build_graph_tensors

config = load_config("code/config/graph_constraints_v2.json")
graph = build_graph_tensors(config)
edge_index = graph["edge_index"]
level_lag_mask = graph["level_lag_mask"]
difference_lag_mask = graph["difference_lag_mask"]
```

`level_lag_mask[e, k] = True` 表示第 `e` 条边允许读取 `X(t-k)`；`difference_lag_mask[e, k] = True` 表示允许读取 `ΔX(t-k)`。这里的 `k=0` 只使用预测起点已经观测到的当前值或当前变化，不是未来数据。

## 下一阶段的正确顺序

1. 先把 0912 数据窗口改成统一的 30 步输出。0912 的 Raw54-GRU 是 15 步（150 s），只能作为开发基线，不能直接进入最终五分钟公平对照表。
2. 实现统一数据接口：用 `t-60..t` 共 61 个原始点构造 `t-59..t` 的 60 点水平历史和 60 点一阶变化历史，并生成未来 30 步 Thv 标签。
3. 主结果先采用 `history_only` 控制模式，不读取预测起点以后的真实阀门值。只有 MPC 在预测前明确给出候选阀门轨迹时，才在另一张表中启用 `known_plan_only` 控制分支。
4. 实现并通过 A0、A2、A6 单种子冒烟测试，重点检查张量形状、lag 索引、序列边界和未来泄漏。
5. 用 3 个种子跑 A0–A7，筛除明显无效或不稳定模块。
6. 对 A0、A3、A4、A5、A6 跑 10 个种子，形成论文主表和消融表；A7 只回答“额外非物理 PCMCI 边是否有增益”。

模型主干是并行结构：Raw54-GRU 全局分支、PCMCI-Lag-GNN 物理分支、三支路 KAN、H_mix KAN 和未来控制编码分支并行产生表示，最后门控融合并一次性输出 30 个 `ΔThv`，与当前 `Thv(t)` 累加得到未来五分钟绝对温度轨迹。

## 论文实验必须保存

- 每个 seed 的整体 RMSE、MAE、R²。
- 50、100、150、200、300 s 各时刻的 RMSE/MAE。
- 快速降温子集、阀门动作子集和完整验证集指标。
- 10 seed 的均值、标准差、95% 置信区间，以及相同样本上的配对误差。
- 参数量、训练时长、峰值显存。
- 验证集逐窗口预测、样本索引、训练/验证曲线。
- 学到的 lag 注意力权重、`alpha_T/alpha_P` 分布和 H_mix KAN 响应曲线。
