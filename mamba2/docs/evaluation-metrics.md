# 评测指标与性能口径

适用于本仓库 `majob.sh` / `demo.py` 和 `eval.sh` / `eval.py`，以当前工作树源码为准。
通用 `litgpt evaluate`、`LLM.benchmark()` 和历史导出文件不自动采用这里的口径。
本页定义指标，不记录未经核验的质量、速度或显存结论。

## KV 槽位压缩率

`eval.sh` 默认同时报告有效 KV 槽位压缩，无需重新训练或增加开关。先用 4 条 32K 样本检查输出：

```bash
DIAG_ARGS='--limit 4 --metadata {"pretrained":"/workspace/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_2 none
```

每次模型调用结束、重置缓存之前统计。令 `S` 为全部层、batch、KV groups 的有效存储条目总数，
包含层级缓存、recent（含注意力层暂存、待提交的尾 token）和 Alpha 精确池（未闭合片段已在池内），排除无效 padding。
`D` 是相同层数、batch、KV groups 下，已进入模型的真实 token 数对应的 dense 槽位总数：

| 输出字段 | 定义 | 含义 |
|---|---|---|
| `kv_retention_ratio` | `S / D` | 保留比例，越低越省 |
| `kv_saving_ratio` | `1 - S / D` | 节省比例，越高越省 |
| `kv_compression_factor` | `D / S` | 压缩倍数，例如保留 25% 即 4 倍 |

控制台打印全部 rank 的汇总；结果 JSON 的 `kv_compression` 保存汇总、按调用类型分组及逐次明细，
同目录的 `*.kv_compression.csv` 保存明细。汇总使用 `ΣS / ΣD`，不直接平均各样本比例；选择题每个
选项、rolling PPL 每个窗口各算一次模型调用。生成任务的分母为截断后的 prompt 加已回填的生成
token 数，**不包含最后一个尚未写入 KV 的输出 token**。这是请求结束时的快照，不是 prefill
结束值或生成期间的平均值；短输入可能尚未发生压缩，需结合明细中的 `processed_tokens` 解读。

此处统计存储条目，`multi` 的虚拟 attention 锚点不重复计数。槽位比例不是显存比例：预分配空位、
元数据、额外 K 副本、打包工作区及临时峰值不在该指标内；原有 `cache_budget` 仍仅报告持久缓冲预算。

## 训练与推理速度

沿用 [README 的训练与评测命令](../README.md#运行入口) 即可采集当前方法指标，无需额外开关。
采集指标不会自动运行 Full KV 对照；比较相对速度仍需同条件下的对照结果。

训练每个 optimizer step 输出实际 token 数对应的全局及平均每卡 tokens/s，以及全局 samples/s、
单步耗时、本次运行累计的有效 step 耗时与累计吞吐；CUDA 峰值显存取所有 rank 中的最大值。
指标写入 TensorBoard（启用时）和 `save_path/training_metrics.jsonl`，后者续训时追加，
`run_start_step` 标记本次启动的起点，`train/run_*` 仅累计本次运行。
token 数覆盖该步全部梯度累积 micro-batch；全局吞吐为各 rank 实际 token 数之和除以最慢
rank 的耗时，平均每卡值再除以卡数。计时同步 GPU，包含取数据、前向、反向和 optimizer，
排除步间日志、checkpoint 和统计汇总；只计完成的 optimizer step。原有各阶段 host 计时
仍只是 CPU 墙钟范围，详细 CUDA 阶段采样仍由 `log_kv_profile_steps` 控制。

推理 JSON 的 `inference_metrics` 和 `*.inference_metrics.xlsx` 保存全部 rank 的明细与汇总：

| 字段 | 口径 |
|---|---|
| `ttft_ms` / `avg_ttft_ms` | 从生成计算开始到首个 token 就绪，含 prefill 和首次采样 |
| `prefill_tokens_per_sec` | 截断后 prompt token 数 / prefill 秒数 |
| `decode_tokens_per_sec` / `decode_ms_per_token` | 首 token 之后的生成速度 / 每 token 耗时 |
| `gen_tokens_per_sec` | 全部生成 token 数 / prefill 与 decode 总秒数 |
| `forward_tokens_per_sec` | 选择题、PPL 的模型输入 token 数 / forward 秒数 |
| `peak_memory_allocated_bytes` / `peak_memory_reserved_bytes` | 每次调用的 CUDA 已分配 / 保留显存峰值 |
| `inference_mlp_chunk_size` | 推理 norm/MLP/残差计算的 token 块大小，0 为整段；记录于 JSON 顶层、`inference_metrics` 和 XLSX 的 `speed_summary` |
| `prompt_len` | 普通生成实际进入模型的 prompt token 数；多轮生成按下文的增量前缀口径 |
| `original_prompt_len` | 普通生成截断前的完整 prompt token 数，包含任务说明和示例 |
| `truncated_prompt_tokens` | 普通生成从 prompt 开头截去的 token 数；与 `prompt_len` 之和为 `original_prompt_len` |

生成计数包含 EOS；只生成一个 token 时没有 decode，相关吞吐和每 token 耗时为 null。
推理计时排除分词、cache 初始化/重置、压缩率统计、评分后处理和文件写入。汇总采用
`Σtokens / Σ阶段秒数`，是汇集各卡请求的**单设备有效吞吐**，不代表集群并行吞吐；
汇总中的秒数也是各请求耗时之和。按 prompt 长度分桶使用相同口径。
显存峰值包含驻留权重、cache 和临时张量，不等于 KV 大小；它是 PyTorch allocator
口径，不含 CUDA context 等外部分配，CPU 运行记为 null。

评测默认使用 2048-token 的 norm/MLP/残差分块，`LOGKV_EVAL_MLP_CHUNK` 可覆盖，0 恢复整段计算；
也可通过 `config_overrides.inference_mlp_chunk_size` 设置，环境变量优先。只在 eval 且无梯度时启用，
不拆分注意力或改变 KV 策略；线性层 GEMM 形状改变可能产生浮点舍入差异。
该执行设置会影响速度和激活峰值，比较 Dense 与 LogKV 时应保持一致，并保留结果中的实际块大小。

多轮 `niah_turns_*` 共用一份 cache，每轮答案单独写入一条完整的生成统计；汇总的
`requests` 和 `avg_ttft_ms` 按答案轮次计数。明细用相同的 `rank`、`sample_id` 关联同一
会话，`turn_index` 从 1 开始，`context_len` 是截断后的共享上下文长度。每轮
`prompt_len` 只计本轮实际送入模型的前缀：首轮为共享上下文加问题，后续轮为上一答案
尚未前向的末 token 加当前问题，按长度分桶也使用此长度。每轮首 token 计入 prefill，
其余生成 token 计入 decode；文本解码与轮间统计不计时。压缩率仍在会话结束时记录一次。

默认保留首次调用/首步，因此初始化、编译预热可能抬高首条耗时；对比稳定速度时可从
明细选取相同长度、预热后的记录。边界同步和跨卡统计有少量开销，不逐 decode token
增加同步，不自动执行额外训练或评测。

## NIAH 任务分数

RULER 的 `niah_*` 任务（含 [`eval_tasks/niah_multi/`](../eval_tasks/niah_multi/) 的多针任务）按请求长度
给分，结果键为长度字符串（如 `32768,none`）：

```text
样本分数 = 输出中出现的参考答案数 / 参考答案数      （转小写后按子串匹配）
长度分数 = 同一 max_length 的样本分数均值
```

- 只计召回、不罚多答：输出多列数字不扣分，含干扰针的 `niah_multikey_*` 需结合样本输出判断。
- RULER 给每条样本补一个 `4096` 键；没有 4096 长度的样本时该值为 `-1`，不是分数。
- `max_length` 是请求长度，样本实际长度在 `length` 字段（prompt token 数加生成预留）。essay 草堆不足时
  `length` 明显短于 `max_length`；多针任务放不下全部针的长度被跳过。两种情况都会告警。

512K/1M 的 RULER 扩展样本另有 `logkv_ruler_protocol`（捞针和逐轮问为 `extended-niah-fit-v1`，
VT 为 `extended-vt-fit-v1`，CWE 为 `extended-cwe-nvidia-v1`，FWE 为 `extended-fwe-fit-v1`，
HotpotQA 为 `extended-qa-fit-v1`）和
`logkv_haystack_repeats`：前者标识扩展协议，后者表示背景词序列重复次数（非 essay 任务为 1）。
`length` 沿用生成器的实际长度口径（包括生成预留），`max_length` 是请求的长度预算；
背景循环发生在插针之前，完整协议与样本数规则见 [512K/1M 说明](eval-512k-1m.md)。

## 输出文件与可追溯性

| 产物 | 内容与限制 |
|---|---|
| `save_path/training_metrics.jsonl` | rank 0 追加写入；记录 step、启动起点、时间、world size、阶段计时与吞吐。`train/run_time_s` 是有效 step 耗时之和，不含启动、步间日志或保存耗时 |
| 评测结果 JSON | `cache_budget`、`kv_compression`、`inference_metrics` 与任务结果；速度明细分为 `generate_until` 和 `loglikelihood`，后者还包含 rolling PPL 调用 |
| `*.kv_compression.csv` | 请求结束时的槽位压缩快照明细 |
| `*.inference_metrics.xlsx` | `speed_summary`；有相应请求时另写 `generate_until`、`gen_by_bucket`、`loglikelihood` |

XLSX 导出依赖 pandas 和可用的 Excel writer；导出失败会告警，不能假定每次都有该文件。
完整速度明细也在 JSON 中。以上文件在本次评测输出路径下生成，不覆盖解释旧文件的需要。

每次比较至少保留代码提交与未提交差异、解析后的 YAML、实际 checkpoint、任务/样本、
截断后长度、生成设置、设备/卡数/精度、预热规则和输出文件路径。
训练日志中的 token 计数是输入张量元素数，不是单独统计的非 padding 或有效 loss token；
`samples` 是各 micro-batch 样本数之和。选择题多个选项和 PPL 多个窗口不能视为独立文档数。

## 三种内存口径

1. **有效槽位**：只反映保留多少条目，不包含预分配空位、元数据、K 副本或工作区。
2. **持久分配字节**：`cache_budget` 统计注册缓存 buffer；SWA 的 `swa_window_size=0`
   按此对齐。详见 [SWA 对照](swa-niah-comparison.md)。
3. **运行峰值**：CUDA allocator peak 包含驻留权重、缓存与调用中的临时张量，
   但不是整个设备的总占用。Alpha 内部预算另计最大可复用 pack 容量，见
   [精确池预算](alpha-logkv.md#同预算与复杂度)。

因此 `kv_saving_ratio=75%` 不能写成“节省 75% 显存”；固定比特数的 KV 量化也不能仅用槽位数比较。

## 新旧结果的区别

旧版本 `decode_ms_per_token` 曾把 prefill 算入总生成耗时；当前定义已改为
`1000 × decode_time_s / decode_tokens`，其中 `decode_tokens=generated_tokens−1`。
不能仅凭列名拼接历史表：应根据代码版本及阶段字段判断，无法确认时保留为旧口径，
不能凭总耗时反推出独立 decode 时间。

当前 JSON 显式保存 `timing_scope`、`throughput_scope`、`memory_scope`。跨 rank 的推理
`Σtokens / Σrequest_seconds` 不代表并行执行的端到端 wall-clock 吞吐；训练全局吞吐则
使用各 rank token 总数除以最慢 rank 的 step 耗时，两者不能直接横比。

底层流式 `generate_fn(timing=...)` 的计时会包含消费者在两次取 token 之间的停顿；
`eval.py` 连续消费生成器。交互输出、网络流式服务或 `LLM.benchmark()` 的数字不能直接
视为相同计时边界。普通生成请求按 EOS 或长度上限完成解码，评分文本截到最先出现的非空
`until` 字符串之前。吞吐与耗时统计包含实际生成的全部 token。

汇总表的 `truthfulqa_gen` 单项保留原始 ROUGE-1 F1（0–100），计算 `AVG_General` 和
`AVG_all` 时先将该项除以 100。

## 实现依据与小检查

- [评测入口](../eval.py)：`cache_compression_snapshot`、`summarize_inference_metrics`、结果汇集与导出。
- [生成计时](../litgpt/generate/base.py)：`generate_fn` 的首 token 和后续 token 时间。
- [训练入口](../demo.py)：`_training_speed_metrics` 与 optimizer step 边界。
- [多针 NIAH 任务](../eval_tasks/niah_multi/niah_multi.py)：生成预留、长度跳过与草堆不足告警；
  [任务测试](../tests/test_niah_multi_tasks.py)覆盖注册、样本生成与召回计分。
- [压缩与速度测试](../tests/test_eval_compression.py)、[多卡汇集测试](../tests/test_eval_collectives.py)、
  [训练计时测试](../tests/test_log_kv_timing.py)。

这些小检查用于核对计数与汇总，不替代 GPU 上的真实速度验证。外部对照的选择与成本边界见
[Baseline 选型记录](baselines.md)。
