本目录的 LitGPT 子集、训练与评测驱动、自定义任务、长文本计划和数据准备工具来自 LogKV 的 niahLogKV 快照。保留上游 Apache-2.0 许可证及各文件原版权声明。

`eval_tasks/` 和 `long_eval_plan.py` 与该快照逐文件相同。适配集中在 `mamba_model.py`、`prepare.py`、`demo.py`、`eval.py`、`litgpt/config.py`、`litgpt/kv_compression.py` 和启动脚本。下载器沿用上游 requests/Hub 策略，支持普通模型文件，并选择 Hub 的普通文件回退模式。表格工具保留上游的列、评分映射和聚合公式，输入改为命令行指定的日志。

SSD 状态转移及门控采用 [Mamba-2](https://arxiv.org/abs/2405.21060)；两级固定分块扫描以 PyTorch 实现。骨干保留 Qwen3 的顺序残差块与 SwiGLU。8B 架构对应 [Qwen/Qwen3-8B-Base 配置](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/config.json)。
