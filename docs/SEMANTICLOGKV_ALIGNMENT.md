# DMS 与 SemanticLogKV 的比较口径

更新：2026-10-15。范围是原始 Base、评测数据、推理与评测接口；保留已经确认的 DMS 官方 adapter-only 训练方法。尚未完成真实 1.7B CUDA 评测。统一规则见 [AGENTS.md](../AGENTS.md) 与 [Baseline 实现规范](BASELINE_IMPLEMENTATION_GUIDE.md)。

## 项目边界与评测入口

**本项目独立运行，不读取、不导入其他实验项目。** 2026-10-15 之前的入口在运行时把相邻 `../litgpt` 加入 `sys.path`、加载其 `eval.py`，并读取其 `exp/` 实验 YAML；这些依赖已全部移除：

| 原依赖 | 现在 |
|---|---|
| `../litgpt/eval.py`、`utils.py`、`litgpt` 包 | 私有快照 `models/_vendor/semanticlogkv/`（命名空间 `models._vendor.semanticlogkv`），来源 `HourunLi/LogKV@4fd97c7`，只保留 `eval.py` 的导入闭包；来源 commit、原始与打包文件 SHA-256、许可证（Apache-2.0）和修改说明见该目录 `SOURCE.json`、`README.md`、`LICENSE.md` |
| `../litgpt/exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml` 继承链 | 项目内平铺快照 `configs/experiment_qwen1.7b_32k.yaml`：原始 Base `ckpt_dir`、`context_length=32768`、仅供 `--backend logkv` 对照使用的 `log_kv_*` 参数；不含 SemanticLogKV CPT 的训练字段 |
| `../litgpt/eval.sh` 任务列表 | 项目内 `configs/eval_suites.yaml`：47 个 main 入口与 3 个 NIAH（仅 `[32768]`） |
| `litgpt.scripts.convert_lit_checkpoint.copy_weights_qwen_3` | `models/qwen3/convert_lit_to_hf.py`（改编自 LitGPT 0.5.12，Apache-2.0），测试中用参考快照的 HF→LitGPT 转换做逐张量零误差往返校验 |

配置加载器只接受项目内的实验 YAML（`--experiment-config` 越界即报错），`config:` 继承不得离开顶层 YAML 所在目录。HF 缓存根目录来自 `DMS_HF_CACHE`（默认仍是服务器共享缓存），在任何 HF 库导入前设置；这些是共享数据资产，不是源码依赖。

`scripts/evaluate.sh` / `majob.sh` → `models.qwen3.evaluate` → 私有加载的快照 `eval.main`。DMS 子类仅覆盖初始化、缓存建立/清理，直接继承 `LogKVLM` 的请求处理、评分、生成与汇集方法；适配只绑定在这个私有模块实例上，不改快照文件，也不替换快照中的全局统计函数。

| 项目 | 对齐方式 |
|---|---|
| 原始权重 | 项目实验 YAML 的 `ckpt_dir/lit_model.pth`，经私有转换器精确转换为 HF；不使用 CPT 权重初始化 DMS |
| 来源验证 | 原权重、源模型配置、tokenizer、实验 YAML、转换代码、输出文件 SHA-256；训练前检查转换产物，评测前再与比较 Base 检查 |
| tokenizer | 双方评测都加载原始 Base 目录的快照 `Tokenizer` |
| 测试缓存 | 与参考相同的缓存根目录及子目录、离线设置、RULER patch；map/filter 结果用 `datasets.disable_caching()` 写进程私有临时文件，多 rank 不竞争共享缓存文件；新版 datasets 写入的不兼容缓存条目明确报错 |
| 任务与长度 | `--suite main`：47 个入口，不传 NIAH metadata（同参考 `eval.sh`）；`--suite niah`：`niah_single_1/2/3`，`metadata.max_seq_lengths=[32768]`；模型窗口取 `context_length=32768`；`metadata.pretrained` 保留原 `ckpt_dir` 字符串 |
| 样本 | 同一 lm-eval（torch218 中为 0.4.11，任务定义随版本变化）、同默认 seeds/fewshot、同 limit；保存 doc/prompt/target hash 与实际 arguments，事后严格核对 |
| 提示与分词 | 不加 chat template；选择题 context 与 continuation 分别编码，continuation `bos=False` |
| 生成 | 共用快照 `generate`；按任务生成参数，非采样时 temperature/top_p=0；在窗口内先预留输出长度，再从左侧截断 prompt |
| 停止 | 共享代码只检查 EOS 与长度，不执行 `until` 字符串；DMS 保持相同行为，不能将其结果宣称为 harness 原生 HFLM 停止规则 |
| 评分 | 同 continuation 对齐位置、fp32 log-softmax、分块累加与 rolling 窗口 |
| 精度与批次 | BF16、eval mode、batch 1；多卡每卡独立模型，按 rank 分配请求 |
| 时间 | 同前向/首 token/decode 计时边界；不同内核、分配行为和编译开销仍可不同，计时一致不等于已经实现同等优化 |
| KV 统计 | 请求结束、reset 前；分母排除最后未 forward 的输出 token；总体使用 Σretained/Σdense；DMS 同时统计 paged 与 contiguous 区 |

DMS 保留原始 KV，不形成合并簇，因此统计中的 `hierarchical_slots=0`；保护窗计入 `recent_slots`，其他保留原始 KV 计入 `exact_slots`。它们是逻辑条目数量，不是分配显存。DMS 缓存动态增长，`persistent_cache_bytes` 为 null；不能把该字段与 LogKV 的常驻预分配预算直接对齐。实际峰值显存沿用共享推理统计（`inference_metrics` 的 `peak_memory_*`）。

## 两处已排除的差异

1. 原始 Base 的 LitGPT 默认窗口是 40960，当前 CPT 保存窗口是 32768。统一入口显式为两者设定 `block_size=32768`，DMS 自身位置窗口不足时直接报错。
2. 参考 `eval.sh` 对 NIAH 默认传六档长度。正式协议只评测 32K：项目任务清单固定 `[32768]`，不能把六档旧结果与本次 32K 混在一起。

## 环境与执行

训练与评测都在 SemanticLogKV 的环境 torch218 中进行，不安装本项目或任何包，从项目根目录运行。训练用本项目的 `dms/training/loop.py` 代替 HF Trainer + accelerate（torch218 没有 accelerate），CPU 上与上游训练逐位一致（见 [验证记录](VALIDATION.md)）；评测入口只检查 `lm-eval>=0.4.9`、`datasets>=4.7`，transformers 4.51.3 缺少的接口由 `dms/transformers_compat.py` 补齐。此前在 transformers 4.57.3 训练环境中导出的模型在 torch218 中的评测行为由 `tests/cross_env_check.py` 核对。私有快照所需的 lightning、jsonargparse、psutil 等 torch218 都已具备；sentencepiece 只在 tokenizer 只有 `tokenizer.model` 时才需要，Qwen3 不需要。两种方法的正式比较在同一评测环境运行，结果记录精确版本，`compare_results` 要求双方一致。

```bash
# torch218，项目根目录：一次性精确转换项目实验 YAML 选中的原始 Base
python -m models.qwen3.prepare_base
# 随后按 README 做训练，得到 student_model（或 bash majob.sh --train configs/qwen3_1.7b_base.yaml 训练后直接评测）

# 只解析配置，不加载权重或数据
python -m models.qwen3.evaluate --model outputs/qwen3_1.7b_base/student_model --suite niah --print-config

# 用同一组 4 个样本先检查真实端到端链路
bash scripts/evaluate.sh outputs/qwen3_1.7b_base/student_model --suite niah --limit 4 --output outputs/compare-dms
bash scripts/evaluate.sh /path/to/semanticlogkv/checkpoint --backend logkv --suite niah --limit 4 --output outputs/compare-semantic
python -m models.qwen3.compare_results outputs/compare-dms/niah/eval_results_TIMESTAMP.json \
  outputs/compare-semantic/niah/eval_results_TIMESTAMP.json
```

`TIMESTAMP` 替换为实际文件名。验证器不比较模型回答是否一致，而是检查比较输入与协议（Base、tokenizer、评测代码、运行时版本含 pyarrow、任务、metadata、limit、窗口、逐样本 prompt/目标/生成参数）是否一致；后端、实际评测权重与预算作为方法差异单独列出。每次启动写 `resolved_eval_<launch_id>.yaml`，不覆盖同目录旧记录；结果 JSON 内保留 `comparison_contract`（含实际评测权重哈希）与 `resolved_eval`（含解释器、CUDA/驱动、GPU、torchrun 拓扑）。

## 仍需区分的实验维度

双方从同一原始 Base 出发，但训练后的参数当然不同。DMS 当前仍采用 OpenR1/8K/冻结底模的适配器蒸馏；SemanticLogKV 当前 CPT 的语料、32K 长度、更新参数量和训练预算与之不同。这次没有将这些训练维度改为相同，不能仅因测试口径对齐就声称训练成本受控。

目前验证证明代码接入、小模型转换与启动器逻辑成立；真实 Base 文件指纹、服务器离线缓存内容、CUDA 内核和正式任务结果，要由目标机器的运行与结果核对确认。优先完成两边相同 4 样本检查，确认可比后再扩大任务数量。
