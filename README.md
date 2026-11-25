# Qwen3-8B-Base + DeepSeek V4-Flash attention

本分支用 Qwen3-8B-Base 初始化 embedding、36 层 MLP、块级 RMSNorm 和 LM head，替换全部注意力后继续训练。36 层沿用 V4-Flash 的配置顺序，前两层为纯滑窗，之后交替 17 个 CSA4 和 17 个 HCA128。所有层的局部窗口均为 128。完整模型有 10,929,049,984 个参数。

训练、权重、任务、预算统计和验证范围见[实现与运行说明](docs/DEEPSEEK4.md)。原 Hybrid-CSA4 项目的平台入口已移至 `scripts/hybrid_csa_majob.sh`，其运行说明见 [Hybrid CSA4](docs/HYBRID_CSA.md)。

```bash
# 无模型、无 GPU，查看完整任务计划及静态缓存预算
python -m deepseek4.evaluate --print-config
python -m deepseek4.budget

# 平台作业每节点执行一次，先准备语料，再运行适配
bash majob.sh prepare
bash majob.sh train

# 先用一个小任务组冒烟，再执行原生主线与 32K–1M 六档
bash majob.sh eval --no-main --lengths 32k --suites ruler --limit 1 --output /path/to/new-smoke
bash majob.sh eval
```
