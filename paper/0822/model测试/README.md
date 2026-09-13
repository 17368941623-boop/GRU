# Thv 第15步模型消融实验

本文件夹用于固定最佳 lookback 后的模型结构与超参数消融。预测量为
`Delta_Thv_15 = Thv(t+15) - Thv(t)`，未来输入只包含
`u(t), ..., u(t+14)` 的八个阀门开度。

## 数据边界

- 训练：从 `train_clean.pkl` 读取后先删除 `0617-ALL`。
- 归一化、快速降温阈值和全部模型参数：仅使用删除 0617 后的训练数据。
- 验证与参数选择：仅使用 Original `260501`。
- 测试：Original `0715-BACK`。
- `train_model_ablation.py`、`run_model_search.py`、
  `select_model_configs.py` 和 `run_selected_configs.py` 不读取任何测试文件。
- 只有生成 `selected_configs.json` 后，`evaluate_selected_models.py` 才读取
  `test_full_clean.pkl`。

## 固定物理图

图节点表示 G1/G2 入口、G 分配点、主路、两个冷屏支路、A 支路、
TE8351、TE8352、TE8353 和 Thv 模组。阀门作为有向边属性：

```text
G1 --CV8300--┐
             ├--> G分配点 --CV8310--> G主路 --┐
G2 --CV8313--┘                                ├--> TE8351 --> TE8352 --> TE8353
             ├--CV8311--> 冷屏支路1          │
             └--CV8312--> 冷屏支路2          │
A  -----------------------CV8351--------------┘

TE8353侧状态 --EC-V2-------> 模组/Thv
             --COOLDOWN----> 模组/Thv
```

代码不创建 KNN 相似度图，也不添加反向边。一次 DAG 拓扑扫描按照上游到
下游的顺序更新节点，因此上游阀门作用可以在一次扫描中传到下游。

## 八类候选模型

1. `gru_baseline`：历史 GRU + 未来阀门编码器。
2. `serial_mlp_gnn_gru`：每个历史时刻先经过 MLP 边门控有向图，再送入 GRU。
3. `serial_kan_gnn_gru`：将本阀门开度的边门控改为单调样条 KAN，再送入 GRU。
4. `parallel_gru_mlp_gnn`：历史 GRU 与当前时刻 MLP-GNN 并行后融合。
5. `parallel_gru_kan_gnn`：历史 GRU 与当前时刻 KAN-GNN 并行后融合。
6. `tcn_baseline`：用因果膨胀 TCN 替代 GRU，检验快速降温趋势是否更适合卷积模型。
7. `serial_mlp_gnn_tcn`：MLP-GNN 图序列后接因果 TCN。
8. `serial_kan_gnn_tcn`：KAN-GNN 图序列后接因果 TCN，可与第7项直接判断TCN路线中的KAN贡献。

KAN只约束“本阀门开度到基础导通能力”为非递减函数。压力、流量、温度和
其他分流阀门通过无单调约束的上下文 MLP 处理，避免施加错误物理约束。

## 参数搜索

`full` 配置包含 57 组结构：

- GNN 隐藏维度：32、64；
- DAG 扫描次数：1、2；
- KAN 网格数：5、8、12；
- TCN 隐藏维度：32、64；
- TCN 层数：根据最佳 lookback 自动测试“刚好覆盖全部历史的层数”及其前一层；
  膨胀率依次为 1、2、4、8、16、32、64，最多 7 层。对于 75–120 步
  lookback，会比较 5 层（感受野 63）和 6 层（感受野 127）。

搜索阶段默认使用种子 42、62，共 114 次训练；每类模型按两个种子的
快速降温验证 RMSE 均值选出一组参数。随后仅对选中参数训练种子 82，
最终三个种子一起测试。`quick` 配置为 19 组结构，适合先检查运行时间。

## 运行

首先在 `残差lookback测试` 完成扩展 lookback 扫描。Mac 终端中：

```bash
cd "/服务器上的0822路径"
bash "残差lookback测试/run_gru_extended_lookback_h15.sh"
```

确定最佳 lookback 后，把下面的 `BEST_LOOKBACK` 替换为最终整数，例如 90：

```bash
bash "model测试/run_all_model_tests.sh" 90 full
```

如果希望无人值守地先完成lookback、自动读取 `260501` 选出的最佳值，再开始
全部模型实验，可以直接在 `0822` 下运行：

```bash
chmod +x "run_complete_thv_h15_pipeline.sh"
bash "run_complete_thv_h15_pipeline.sh" full
```

脚本严格顺序运行，一个训练正常结束后才开始下一个。某一步报错时立即停止；
重新执行会跳过已有的 `metrics.json`。需要覆盖已有结果时：

```bash
OVERWRITE=1 bash "model测试/run_all_model_tests.sh" 90 full
```

可以通过环境变量调整运行参数，例如：

```bash
EPOCHS=120 BATCH_SIZE=512 NUM_WORKERS=0 \
bash "model测试/run_all_model_tests.sh" 90 full
```

Apple MPS 不建议同时启动多个训练进程，因此脚本没有后台并行。
正式训练前，脚本会先用合成张量检查八类模型的前向传播、反向传播和
KAN单调性；这个检查不读取任何项目数据。

## 主要输出

```text
outputs/
├── development/horizon_15/lookback_XX/
│   ├── search_manifest.csv
│   ├── development_all_runs.csv
│   ├── development_config_summary.csv
│   ├── selected_configs.json
│   └── 模型名/配置名/seed_XX/
│       ├── best_model.pt
│       ├── metrics.json
│       ├── training_history.csv
│       ├── validation_predictions.csv
│       ├── training_curve.png
│       └── validation_prediction.png
└── final_test/horizon_15/lookback_XX/
    ├── final_all_seed_runs.csv
    ├── final_model_summary.csv
    ├── final_model_comparison.png
    └── final_evaluation_audit.json
```

最终表按验证集快速降温 RMSE 排序，不使用测试集重新排序或改变参数。
