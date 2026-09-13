# 五终端并行模型实验

本目录固定用于 `lookback=60`、预测未来第15步、`full` 参数搜索。每个分片脚本
内部始终顺序训练，因此同时打开五个终端时最多有五个训练进程。不同分片使用
互斥的 `(模型配置, seed)`，不会写入同一个运行目录。

## 第一阶段：参数搜索

在五个终端中分别运行：

```bash
bash "并行脚本_5终端/01_search_part_1.sh"
bash "并行脚本_5终端/01_search_part_2.sh"
bash "并行脚本_5终端/01_search_part_3.sh"
bash "并行脚本_5终端/01_search_part_4.sh"
bash "并行脚本_5终端/01_search_part_5.sh"
```

114次搜索训练被分为 `23 / 23 / 23 / 23 / 22` 次。必须等五个终端全部出现
`SEARCH_SHARD_RUNS_COMPLETE` 后再进入第二阶段。某个分片中断时，只需重新运行
对应脚本；已有 `metrics.json` 的任务会自动跳过。

## 第二阶段：验证集选择配置

只在一个终端运行：

```bash
bash "并行脚本_5终端/02_select_configs.sh"
```

该步骤不训练，也不读取0715-BACK。若任何搜索任务缺失，程序会拒绝继续并列出
未完成配置。

## 第三阶段：种子82确认

再次在五个终端中分别运行：

```bash
bash "并行脚本_5终端/03_confirmation_part_1.sh"
bash "并行脚本_5终端/03_confirmation_part_2.sh"
bash "并行脚本_5终端/03_confirmation_part_3.sh"
bash "并行脚本_5终端/03_confirmation_part_4.sh"
bash "并行脚本_5终端/03_confirmation_part_5.sh"
```

八个选中模型按 `2 / 2 / 2 / 1 / 1` 分配。必须等五个终端全部完成后再打开
测试集。

## 第四阶段：最终测试

只在一个终端运行：

```bash
bash "并行脚本_5终端/04_final_test.sh"
```

这一步才读取Original 0715-BACK，并对八类验证集选中模型的42、62、82三个种子
进行推理。测试结果不重新改变配置或排序。

## 注意

- 所有命令均从 `model测试` 目录运行。
- 不要同时运行旧的 `run_all_model_tests.sh`，否则会与分片任务重复。
- 默认不会覆盖已完成结果；不要设置 `MODEL_OVERWRITE=1`。
- 日志保存在 `model测试/logs_h15_lb60_parallel/`。
