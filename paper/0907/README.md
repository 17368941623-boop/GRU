# 0907 Thv物理特征消融训练包

本目录可整体搬到Mac服务器。实验固定使用并行GRU-KAN-GNN，预测未来第15步的Thv；历史窗口60步，公共预测起点要求120步。与0906架构比较保持相同的数据划分、损失、隐藏层规模和评价规则。

## 运行

进入0907目录并激活已有PyTorch环境：

```bash
bash run_feature_ablation_0907.sh
```

默认同时运行5个训练进程。服务器资源允许时可改为8或10个：

```bash
WORKERS=8 bash run_feature_ablation_0907.sh
```

如需先完成训练和验证、暂不读取0715-BACK：

```bash
RUN_FULL_TEST=0 bash run_feature_ablation_0907.sh
```

脚本支持断点续跑；再次执行会复用身份和协议均匹配的完整结果。不要同时启动两个总脚本写入同一个结果目录。

## 固定实验设计

- 目标：Delta Thv(t+15)，计算指标前恢复为绝对Thv(t+15)。
- 历史：截至t的原始量与当前消融允许的因果构造特征。
- 未来：仅8个原始阀门指令u(t)至u(t+14)；未来温度、压力、流量和未来构造特征均不输入。
- 训练：排除0617-ALL；Standardizer和快速降温阈值只拟合训练集。
- 验证：仅Original 260501用于早停、学习率调度和检查点冻结，主指标为完整验证RMSE。
- 测试：100次训练全部完成且验证汇总冻结后，才允许读取Original 0715-BACK。它是降温后复温、阀门动作较丰富的跨工况事后测试，不称为未经查看的最终独立测试。
- 损失：训练集中Delta最低10%的样本使用3倍权重的Huber损失。
- 架构：并行GRU与固定有向管路GNN；KAN只保证各阀门边导通函数对自身开度非递减，不保证最终温度对阀门全局单调。

## 十组特征配置

共10组配置乘10个相同随机种子，合计100次训练：

1. raw20：仅20个原始历史量。
2. full20：原始量加全部20个推荐构造特征。
3. full20_no_target_history_dynamics：移除目标温度短差分和长趋势。
4. full20_no_g_main_flow_command：移除G1/G2串联阀门与压差导通代理。
5. full20_no_premix_postmix_thermal_drive：移除G1/G2/A混气热驱动与CV8351动作。
6. full20_no_postmix_cooling_bridge：移除TE8351处CoolProp比焓冷却尺度与流量加权桥接。
7. full20_no_mainline_state_contrasts：移除TE8351至TE8353温差和压力差。
8. full20_no_thv_module_coupling：移除模组入口温差、流量温差驱动和状态机温差。
9. full20_no_module_valve_dynamics：移除EC-V2和COOLDOWN动作差分。
10. full22_with_apparent_heat_leak：在full20上增加两项可选CoolProp表观焓增。

为避免参数量混杂，所有配置都保留相同的42维历史输入和完全相同的网络参数量。被消融的构造特征在使用训练集统计量标准化后置零。raw20会屏蔽全部22个构造特征；full20只屏蔽两项可选表观焓增。

## 主要输出

- outputs_0907/development：每个配置和种子的权重、Scaler、逐轮训练记录、验证预测和图。
- validation_summary/validation_feature_summary.csv：验证集均值、标准差和95%置信区间。
- validation_summary/paired_vs_raw20.csv：相同种子下各配置相对原始输入的差异。
- validation_summary/paired_component_effects.csv：逐组移除相对full20的配对差异。正差值表示移除后更差，即该组在full20背景下有正贡献。
- full_test_summary：相同结构的完整0715-BACK事后测试结果。

特征是否保留应先按验证结果确定，再用0715-BACK判断跨工况结论是否一致。若某组在验证和测试中的方向相反，应报告其工况依赖性，不能只选择更好看的测试结果。
