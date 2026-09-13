# GRU — 物理约束 GRU / GRU-ODE 低温测试站温度预测

本仓库汇总低温测试站温度预测研究的**代码、文档与图表**。
原始数据集、模型权重等大体积文件未纳入 Git（见下方“未入库内容”）。

## 目录结构

```
GRU/
├── new_data/          数据构建、参数搜索与因果图分析（代码/文档/图表）
│   ├── processed_data/        数据构建配置与特征目录
│   ├── Template do not modify/ 实验模板
│   ├── test1 big hotmap/       lookback / hidden size 大范围热图
│   ├── test2 small hotmap/     lookback / hidden size 细范围搜索
│   ├── test3 PCMCI计算/        PCMCI 因果分析
│   ├── 中间文件/               静态物理图构建脚本与预览
│   ├── readme.md               论文结构说明
│   └── 验证集RMSE计算方式.md     验证集困难降温片段 RMSE 评价口径
└── paper/             模型、实验与论文材料
    ├── model-F-GRU/            GRU 基础模型与训练代码
    ├── model-F-GRU-ODE/        结合 ODE 的改进版本
    ├── 论文中文初稿_0909/       中文初稿 (LaTeX 源码 + 编译 PDF)
    ├── els-cas-templates/      Elsevier LaTeX 模板
    ├── 0718all … 0912/         各日期实验输出与图表
    ├── analysis/               分析脚本
    ├── figure_work_0904/       图表工作目录
    ├── industrial-data-viewer/ 工业数据可视化工具（源码）
    ├── 图片/                    论文图片
    ├── *.docx / *.pptx         说明文档、消融实验报告、汇报材料
    └── readme.txt              模型文件简要说明
```

## 核心评价口径

预测任务为 10 s 采样、15 步（150 s）预测，目标为 `Delta Thv(t+15)`，评价恢复后的绝对温度
`Thv(t+15)`。主排名指标为 260428 验证集四个困难降温片段 (H1–H4，共 1440 点) 的五种子
**汇总 RMSE**（先合并误差再开方，而非片段 RMSE 平均）。详见
`new_data/验证集RMSE计算方式.md`。

## 未入库内容（因体积超限，仅保存在本地）

- 原始与处理后的数据集：`*.csv`（约 12 GB）、`processed_data/*.pkl`、`*_raw54.pkl` 等
- 模型权重与训练产物：`*.pt`、`saved_models_*`
- 归档与依赖：`*.zip`、`node_modules/`、`__pycache__/`、`__MACOSX/` 等

上述文件已由 `.gitignore` 排除。如需完整数据，请联系仓库所有者。

## 内部资料

- `new_data/readme.md`：论文整体结构与实验规划
- `paper/论文中文初稿_0909/初稿说明与证据边界.md`：初稿说明与证据边界
- `paper/消融实验报告 v1.0.docx`：消融实验报告
