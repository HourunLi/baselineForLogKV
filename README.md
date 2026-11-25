# Qwen3-8B-Base + DeepSeek V4-Flash attention

本分支用 Qwen3-8B-Base 初始化 embedding、36 层 MLP、块级 RMSNorm 和 LM head，替换全部注意力后继续训练。36 层沿用 V4-Flash 的配置顺序，前两层为纯滑窗，之后交替 17 个 CSA4 和 17 个 HCA128。所有层的局部窗口均为 128。完整模型有 10,929,049,984 个参数。

训练、权重、任务、预算统计和验证范围见[实现与运行说明](docs/DEEPSEEK4.md)。原 Hybrid-CSA4 项目的平台入口已移至 `scripts/hybrid_csa_majob.sh`，其运行说明见 [Hybrid CSA4](docs/HYBRID_CSA.md)。

在已有的 8B `niahLogKV` 平台作业环境中，切到 `deepseek4` 分支，每节点运行一次：

```bash
bash majob.sh all
```

入口继承参照作业的 `LONG_PPL_DATA`、`LONG_PPL_BOOKS` 和其余离线数据设置，按顺序执行环境与数据检查、对照作业的 ProLong 分片核验、全模型 CPT、32K 主线和 32K–1M 六档评测。完整模式要求全部受支持的套件，任一阶段失败即停止。默认配置与 8B LogKV 对齐：32,768 上下文、全局 batch 128、2,000 次优化器更新，共 8,388,608,000 个训练 token。8 卡时每卡 micro-batch 为 1，累积 16 次。训练每 400 步保存可恢复断点，中断后用同一命令和输出目录续训。

```bash
# 仅查看完整任务计划，无需 GPU 或权重
python -m deepseek4.evaluate --full --print-config
python -m deepseek4.budget

# 在平台上只检查环境与数据
bash majob.sh check
```
