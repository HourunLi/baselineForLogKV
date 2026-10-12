# DMS 与 SemanticLogKV 的比较口径

更新：2026-10-12。此次范围是原始 Base、测试数据、推理与评测接口；保留已经确认的 DMS 官方 adapter-only 训练方法。尚未完成真实 1.7B CUDA 评测。

## 仓库边界与评测入口

源码基准为相邻 `litgpt` 仓库根目录 `eval.py`，不是 `litgpt/eval/evaluate.py`。当前默认实验配置为 `exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml`，递归继承设置沿用该仓库的 YAML loader 与环境变量展开。

**所有 baseline 适配只发生在本仓库，`litgpt` 保持只读。** 不往参考仓库添加模型分支、注册机制、指标接点或 baseline 文档。本次早先添加到 `eval.py`、`litgpt/kv_compression.py`、`docs/evaluation-metrics.md` 的改动已全部撤回，既有其他文件改动保留。

`scripts/evaluate.sh` → 本仓库 `models.qwen3.evaluate` → 私有加载的原版 `eval.main`。本仓库的 DMS 子类仅覆盖初始化、缓存建立/清理，直接继承 `LogKVLM` 的请求处理、评分、生成与汇集方法。在该私有模块实例内选择 DMS 构造类和统计适配函数；不改参考文件，也不替换全局 `litgpt.kv_compression` 函数。原生评测实例和原有命令仍使用原来的 LogKV 类。

启动器在自身输出目录调用原入口，避免其 `eval_results_cache.json` 写入参考目录；导入时禁用字节码写入，拒绝把输出目录放入 `litgpt`。来源信息由 baseline 在本次新结果中补充，DMS 结果移除不适用的 `log_kv_*` 参数。新增 baseline 应继续在本仓库兼容这一接口，不要求参考实现认识外部方法。

| 项目 | 对齐方式 |
|---|---|
| 原始权重 | 使用训练配置 `ckpt_dir/lit_model.pth`，用该仓库 `copy_weights_qwen_3` 精确转换为 HF；不使用 CPT 权重初始化 DMS |
| 来源验证 | 原权重、源模型配置、tokenizer、转换代码、输出文件 SHA-256；训练前检查转换产物，评测前再与比较 Base 检查 |
| tokenizer | 双方评测都加载原始 Base 目录的 `litgpt.tokenizer.Tokenizer` |
| 测试缓存 | 共用 `eval.py` 中 `/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache` 及子目录、离线设置、RULER patch |
| 任务与长度 | 默认 `niah_single_2,niah_single_3`；模型窗口和 `metadata.max_seq_lengths` 均取训练配置 `context_length=32768`；RULER `metadata.pretrained` 保留原 `ckpt_dir` 路径字符串 |
| 样本 | 同 harness 0.4.9、同默认 seeds/fewshot、同 limit；保存 doc/prompt/target hash 与实际 arguments，事后严格核对 |
| 提示与分词 | 不加 chat template；选择题 context 与 continuation 分别编码，continuation `bos=False` |
| 生成 | 共用 `litgpt.generate.base.generate`；按任务生成参数，非采样时 temperature/top_p=0；在窗口内先预留输出长度，再从左侧截断 prompt |
| 停止 | 目前共享代码只检查 EOS 与长度，不执行 `until` 字符串；DMS 保持相同行为，不能将其结果宣称为 harness 原生 HFLM 停止规则 |
| 评分 | 同 continuation 对齐位置、fp32 log-softmax、分块累加与 rolling 窗口 |
| 精度与批次 | BF16、eval mode、batch 1；多卡每卡独立模型，按 rank 分配请求 |
| 时间 | 同前向/首 token/decode 计时边界；不同内核、分配行为和编译开销仍可不同，计时一致不等于已经实现同等优化 |
| KV 统计 | 请求结束、reset 前；分母排除最后未 forward 的输出 token；总体使用 Σretained/Σdense；DMS 同时统计 paged 与 contiguous 区 |

DMS 保留原始 KV，不形成合并簇，因此统计中的 `hierarchical_slots=0`；保护窗计入 `recent_slots`，其他保留原始 KV 计入 `exact_slots`。它们是逻辑条目数量，不是分配显存。DMS 缓存动态增长，`persistent_cache_bytes` 暂为 null；不能把该字段与 LogKV 的常驻预分配预算直接对齐。实际峰值显存沿用共享推理统计。

## 两处已排除的差异

1. 原始 Base 的 LitGPT 默认窗口是 40960，当前 CPT 保存窗口是 32768。统一入口显式为两者设定 `block_size=32768`，DMS 自身位置窗口不足时直接报错。
2. 原 `eval.sh` 对 NIAH 默认传六档长度，而且不直接转发 YAML metadata/limit。当前 README 的 Alpha/Beta 命令通过 DIAG_ARGS 选单档 32K。统一入口明确生成单档 metadata，并保存解析配置；不能把六档旧结果与本次 32K 混在一起。

## 环境与执行

仅 baseline 仓库需要这些适配修改，参考 `litgpt` 使用原有实现。将两个仓库保持相邻，或通过 `--litgpt-root /path/to/litgpt` 指定。默认 Base/缓存路径来自服务器挂载；本地仓库不包含真实权重或测试数据。

训练环境：`pip install -e '.[train,test]'`；评测环境：`pip install -e '.[eval,test]'`。两者都安装 `pip install -e ../litgpt` 核心包。训练保留 datasets 4.4.2；评测采用 SemanticLogKV 所需 datasets<4（本地验证3.6.0）和 lm-eval0.4.9。不要将 train/eval extras 合并安装，也不要安装 LitGPT `[extra]` 带入冲突的 Transformers 上限。两种方法的正式比较应在同一评测环境运行。

```bash
# 训练环境：一次性精确转换当前配置选中的原始 Base
python -m models.qwen3.prepare_base
# 随后按 README 做训练，得到 student_model

# 评测环境：只解析配置，不加载权重或数据
python -m models.qwen3.evaluate --model outputs/qwen3_1.7b_base/student_model --print-config

# 用同一组4个样本先检查真实端到端链路
bash scripts/evaluate.sh outputs/qwen3_1.7b_base/student_model --limit 4 --output outputs/compare-dms
bash scripts/evaluate.sh /path/to/semanticlogkv/checkpoint --backend logkv --limit 4 --output outputs/compare-semantic
python -m models.qwen3.compare_results outputs/compare-dms/eval_results_TIMESTAMP.json outputs/compare-semantic/eval_results_TIMESTAMP.json
```

`TIMESTAMP` 替换为实际文件名。验证器不比较模型回答是否一致，而是检查比较输入是否一致。输出目录中还有 `resolved_eval.yaml`，分别保存 baseline 后端、传给原接口的参数及比较条件；每份结果 JSON 内保留独立 `comparison_contract`，包含参考代码和兼容代码哈希及依赖版本。原 `litgpt/eval.py` 不支持也不需要 `--model_backend dms`；DMS 始终从本仓库入口运行。

## 仍需区分的实验维度

双方从同一原始 Base 出发，但训练后的参数当然不同。DMS 当前仍采用 OpenR1/8K/冻结底模的适配器蒸馏；SemanticLogKV 当前 CPT 的语料、32K 长度、更新参数量和训练预算与之不同。这次没有将这些训练维度改为相同，不能仅因测试口径对齐就声称训练成本受控。

目前验证证明代码接入与小模型转换成立；真实 Base 文件指纹、服务器离线缓存内容、CUDA 内核和正式任务结果，要由目标机器的运行与结果核对确认。优先完成两边相同4样本检查，确认可比后再扩大任务数量。
