# Raw54 PCMCI 计算说明

## 目录与数据边界

服务器上保持以下相邻目录结构：

```text
工作目录/
├── processed_data/
│   ├── train_clean.pkl
│   └── feature_catalog.json
└── test2 PCMCI计算/
    ├── code/
    ├── output/
    └── output_picture/
```

代码只读取 `processed_data/train_clean.pkl` 和 `feature_catalog.json`。只使用 `source_variant == Original` 的实测训练数据；不读取扩增样本、验证集和测试集。完整的 `0617-ALL` 会被纳入。54 个变量均来自 `raw_signal_columns`，构造特征不进入 PCMCI。

## 一键运行

```bash
cd "test2 PCMCI计算"
bash code/run_pcmci.sh
```

脚本优先用 `conda run -n pytorch python`。如果服务器环境名不同：

```bash
CONDA_ENV_NAME=你的环境名 bash code/run_pcmci.sh
```

默认运行两套完整 Raw54 PCMCI，并额外运行 H_mix 专项核查：

1. `difference`：主分析。对每个 Raw54 信号作一阶差分，问题是“X 的变化在若干步后是否解释 Y 的变化”。
2. `level`：敏感性分析。使用原始数值水平，问题是“X 的数值水平在若干步后是否解释 Y 的数值水平”。
3. `hmix_diagnostics`：用全部 Original 工况计算正负向差分相关和阀门事件响应，专门识别搜索边界假象、控制策略和阀门联动。
4. `hmix_targeted`：只对 `TE8352/TE8353/A管/EC-V2/COOLDOWN/FC-V1 -> Thv` 做预先指定的10–600秒条件时延检验，所有360个目标滞后都会执行bootstrap。

只跑主分析可用：

```bash
RUN_LEVEL_SENSITIVITY=0 bash code/run_pcmci.sh
```

快速检查可降低 bootstrap 次数：

```bash
BOOTSTRAP=5 RUN_LEVEL_SENSITIVITY=0 bash code/run_pcmci.sh
```

正式结果建议保留默认 `BOOTSTRAP=50`。`MAX_BOOTSTRAP_CANDIDATES=0` 表示不限制统计bootstrap候选；`MAX_EXPORT_EDGES=200` 只控制紧凑GNN图的大小，不再截断统计检验。H_mix专项默认 `HMIX_MAX_LAG=60`（600秒）并使用每个片段的全部数据；如服务器内存受限，可设置 `HMIX_MAX_SAMPLES_PER_SEGMENT=3000`。

## 计算流程

1. 以 7 个 Original 工况为独立父工况，并保留其 41 个连续片段边界。
2. 固定随机种子后，把父工况拆成互不重叠的“相关性筛选组”和“PCMCI 推断组”。
3. 筛选组计算全部 `54 × 54 × 30 = 87,480` 个滞后 Pearson 相关系数。
4. 对每个非控制目标变量，保留绝对相关性最高且达到阈值的 30 个跨变量 `(来源, 目标, 滞后)` 候选。30 是每个目标总计 30 条，不是每一对变量 30 条。
5. 将这些数据候选与静态物理图规定的全部允许滞后候选取并集；同时为每个非控制变量加入自身 1–30 步历史作为“自回归条件项”，防止慢变化序列因没有控制自身历史而产生虚假跨变量边。自回归项不占用每目标 30 条跨变量候选名额，也不冒充静态物理边。
6. 在另一组父工况上只对上述候选并集执行 PC1 和 MCI 条件独立检验。
7. 对候选并集的 MCI p 值作 Benjamini-Hochberg FDR 校正，得到 q 值。
8. 以整个父工况为单位重采样，进行固定父集合的 MCI bootstrap，保留方向/符号稳定的边。所有通过q值、效应阈值及每对最多3个滞后限制的候选都会执行bootstrap；不会再因为最终GNN只导出200条而漏检物理候选。

主动控制阀被当作外生输入，代码禁止学习“过程变量指向未来阀门开度”的交叉边。`H_mix` 是虚拟汇合状态，数据中没有这个传感器，因此 PCMCI 不会伪造该节点，而是输出 `A管/EC-V2/COOLDOWN/FC-V1 -> Thv` 的代理时延，供后续 H_mix KAN 模块使用。

## 输出文件

- `output/preflight.json`：数据量、父工况、Raw54、预期 87,480 个相关计算等预检结果。
- `output/difference/lagged_correlation_screen.csv`：全部 87,480 个差分滞后相关。
- `output/difference/pcmci_candidate_inventory.csv`：进入 PC1/MCI 的候选及其来源。
- `output/difference/pcmci_candidate_tests.csv`：每个候选的效应、p、q、bootstrap 稳定性。
- `output/difference/stable_pcmci_edges_all.csv`：完成bootstrap后的全部稳定差分边，作为统计审计依据。
- `output/difference/stable_pcmci_edges.csv`：按分数截取的紧凑图，默认最多200条，仅供建模候选。
- `output/difference/stable_physical_lags.csv`：全部稳定物理先验时延，不受紧凑图200条上限影响。
- `output/difference/stable_lagged_graph_all.json`：全部稳定边的机器可读图。
- `output/difference/stable_lagged_graph.json`：紧凑GNN候选图。
- `output/difference/latent_H_mix_proxy_lags.csv`：虚拟 H_mix 模块的代理时延。
- `output/difference/latent_H_mix_candidate_tests.csv`：H_mix全部1–30步候选，包括未通过效应或稳定性阈值的结果。
- `output/difference/unsupported_static_candidates.csv`：物理先验中未获当前数据支持的候选；这不等于物理关系不存在。
- `output/level/`：同样结构的原始水平敏感性结果。
- `output/difference_vs_level_edges.csv` 与 `difference_vs_level_summary.json`：两种变换的重合情况。
- `output/hmix_diagnostics/`：10–600秒正负向差分相关、事件响应、阀门之间的水平/动作相关。
- `output/hmix_targeted/difference/` 和 `level/`：H_mix六个预先指定来源到Thv的360个条件滞后检验和全部bootstrap结果。

## H_mix边界结果的解释规则

`lag=30` 或 `lag=60` 处显著不代表“真实响应必然超过搜索边界”。只有当以下证据一致时，才把它解释为主要物理时延：

1. PCMCI/MCI效应通过FDR、效应阈值和父工况bootstrap；
2. 全部工况和多数单独工况的正向lead/lag峰值落在相近区间；
3. 阀门事件后Thv响应也出现在相近区间；
4. 阀门动作与其他阀门的共线性不会完全解释该结果；
5. 峰值不是只贴在搜索上界且继续向上增长。

因此，`hmix_targeted` 给出的是条件预测时延，`hmix_diagnostics` 给出事件和方向核查；二者应联合判读。

服务器计算结束后，请把整个 `output` 目录原样复制回来，不要只复制 `stable_pcmci_edges.csv`，否则无法审计候选筛选、多重比较和稳定性。

## 后续进入 GNN 时的读取语义

若边为 `X(t-τ) -> Y(t)`：

- `difference` 图对应读取 `ΔX(t-τ) = X(t-τ) - X(t-τ-1)`；
- `level` 图对应读取 `X(t-τ)`；
- 边的 `lag_steps` 乘以 10 即秒数；
- 不应把 `difference` 发现的边悄悄改成读取原始水平，否则因果筛选问题和模型输入语义不一致。
