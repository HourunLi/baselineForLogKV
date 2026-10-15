# PyramidKV baseline（Qwen3-1.7B-Base，SemanticLogKV 评测口径）

本分支是一个**独立项目**：在 Qwen3 上实现 PyramidKV，并按 SemanticLogKV 的数据集、请求接口和指标口径评测，用于横向对比。

- PyramidKV 的选择规则移植自 [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)（MIT），见 [LICENSE](LICENSE)。
- 评测侧的模型、tokenizer、KV cache、生成、打分、指标和结果格式全部在本仓库实现。**不 import、不加载 litgpt / SemanticLogKV 的任何代码**，只依赖 PyTorch、tokenizers、lm-eval 等通用库。
- 与 SemanticLogKV 对齐的内容是“行为”：同一份原始 Base 权重、同一 tokenizer 规则、同一 lm-eval 任务与缓存、同样的截断/生成/打分规则、同样的 KV 槽位与速度指标定义、同样的结果文件布局。

## KVCache-Factory 是怎么比较不同方法的

KVCache-Factory 把多种 KV 压缩方法放在同一个推理入口下，用参数切换：

| 组件 | 作用 |
|---|---|
| `pyramidkv/monkeypatch.py` 的 `replace_llama(method)` / `replace_mistral(method)` | 按 `--method` 把 HF `LlamaAttention.forward` 等替换成对应方法（`pyramidkv`、`snapkv`、`h2o`、`streamingllm`、`fullkv` 等） |
| `pyramidkv/pyramidkv_utils.py` 的 `*KVCluster.update_kv` | 每种方法的核心：prefill 时用完整 KV 计算注意力，然后只把选中的 KV 写进 cache；decode 时在压缩后的 cache 上追加 |
| `run_longbench.py` / `run_needle_in_haystack.py` / `run_ruler.py` | 三个评测 runner；`--max_capacity_prompts` 设定每层预算，`--kv_cache_granularity`、`--gqa_score_agg` 控制 GQA 布局 |
| `scripts/scripts_*/eval.sh`、`eval_methods.sh`、`metrics.sh` | 单次运行、预算扫描（64…4096）与打分汇总 |

例如 `python run_longbench.py --method pyramidkv --model_path ... --max_capacity_prompts 128 --attn_implementation flash_attention_2`；换成 `--method h2o` 即比较 H2O。

不能直接拿它和 SemanticLogKV 比：它只 patch Llama/Mistral（锁定 transformers 4.44.2，没有 Qwen3 的 q/k norm），其 RULER/LongBench 数据、prompt 模板、HF `generate` 停止规则和打分都与 SemanticLogKV 不同。因此本分支只移植 PyramidKV 的算法核心，再按 SemanticLogKV 的接口自己实现评测。

## 方法实现

每层的流程与 KVCache-Factory 相同：

1. **Prefill**：用整段 prompt 的完整 K/V 做因果注意力（prefill 输出不受压缩影响）。
2. **评分**：最后 `window_size` 个 query 对更早的 key 做 softmax（窗口内因果），在窗口上求和，再做 1-D pooling（SnapKV 观察窗口）。
3. **金字塔预算**：低层多、高层少，`beta` 控制坡度；各层平均约为 `max_capacity_prompt`（含窗口）。prompt 短于预算时不压缩；略长于预算时退化为各层相同预算。
4. **真实删除**：每个 head 保留 top-k 旧 token 和最近窗口，其余 K/V 不再存储。
5. **Decode / 续写**：新 token 追加到压缩后的 cache，不再淘汰（KVCache-Factory 语义）。

| 参数 | 默认值 | 来源 |
|---|---|---|
| `max_capacity_prompt` | 2048 | KVCache-Factory 初始化默认；论文报告 128 与 2048 |
| `window_size` / `kernel_size` / `pooling` | 8 / 7 / `maxpool` | `run_longbench.py`、`run_ruler.py` 对 `pyramidkv` 的设置 |
| `beta` | 20 | `PyramidKVCluster` 默认 |
| `kv_cache_granularity` | `kv_head` | 每个 KV head 一份条目，与 SemanticLogKV 的 KV group 槽位同单位；`query_head` 复现旧布局（GQA 下条目翻倍） |
| `gqa_score_agg` | `mean` | 组内 query head 分数聚合 |

`kv_head` 下的打分等价于 KVCache-Factory 先 `repeat_kv` 再聚合的写法，只是不实际复制张量；测试中与其原始代码逐元素比较。Qwen3 模型本身与 Transformers 的 `Qwen3ForCausalLM` 做了数值对照。

**选择题/PPL 打分**（`--loglikelihood-mode`）：PyramidKV 只在一次前向结束后压缩，所以同一次前向内的 token 看到的都是完整 KV。默认 `compress_context` 先 prefill 并压缩 context，再在压缩后的 cache 上为 continuation 打分（与生成答案时的处境一致）；`single_forward` 一次前向打完分，结果等同 FullKV。rolling PPL 的 context 只有 1 个 token，不会触发压缩。NIAH 是生成任务，不受此选项影响。

## 与 SemanticLogKV 的对齐

| 项目 | 本分支做法 |
|---|---|
| 权重 | 直接读取 SemanticLogKV 实验 `ckpt_dir` 下的原始 `lit_model.pth`（参数名按 LitGPT 布局，无需转换），也支持 HF safetensors；严格加载，缺/多张量即报错 |
| tokenizer | 自行实现与参考评测相同的规则：BOS/EOS 来自 `tokenizer_config.json` 与 `generation_config.json`，自动去除后端多加的 BOS/EOS，不加 chat template |
| 任务与数据 | lm-eval 0.4.9；默认 `niah_single_2,niah_single_3`；`metadata.max_seq_lengths=[32768]`，`metadata.pretrained` 原样使用 Base 路径字符串；HF/RULER 离线缓存根目录相同 |
| 截断 | 打分时左截断 context；生成时先预留 `max_gen_toks` 再左截断 prompt；空 context 用 `[bos]` |
| 生成 | 非采样时 temperature/top_p 置 0 走 argmax；只在 EOS 或长度上限停止，不执行 `until` 字符串（与参考评测一致） |
| 打分 | continuation `bos=False` 单独编码；fp32 log-softmax，1024 位置分块；rolling 窗口 `max_seq_length-1` |
| KV 指标 | 请求结束、释放 cache 前快照；分母为已前向 token × 层 × KV group，不含最后一个未回填的输出 token；汇总用 ΣS/ΣD；`recent`=窗口+追加 token，`exact`=选中的旧 token，`hierarchical`=0 |
| 速度/显存 | 同样的 prefill（含首 token 采样，PyramidKV 压缩计入 prefill）/decode 计时边界与每请求 CUDA 峰值 |
| 多卡 | 每卡一份模型，按 `requests[rank::world]` 分配，Gloo 汇总后按原顺序重排 |
| 输出 | `eval_results_<时间>.json`（`cache_budget`、`kv_compression`、`inference_metrics`、`metadata`、`results`，含 lm-eval 逐样本日志）及 `.kv_compression.csv`、`.inference_metrics.xlsx`、`.csv`、`resolved_eval.yaml` |

`cache_budget.persistent_cache_bytes` 为 `null`：cache 按需增长，没有 SemanticLogKV 那样的预分配常驻缓冲；逐请求的 `retained_kv_bytes` 给出保留条目的 K/V 字节数（不含预留空位与临时打分工作区）。

## 安装

建议与 SemanticLogKV 评测使用同一个 CUDA 环境（相同的 torch、lm-eval、datasets 版本）：

```bash
git clone -b PyramidKV https://github.com/HourunLi/baselineForLogKV.git PyramidKV
cd PyramidKV
pip install -e '.[eval,test]'
```

lm-eval 0.4.9 与 transformers 5.x 不兼容，`[eval]` 已限制 `transformers<5`。prefill 使用 PyTorch SDPA，不依赖 FlashAttention 包。

## 运行方式

以下命令都在仓库根目录执行。

```bash
# 0) 只解析配置，不加载权重和数据
python -m pyramidkv.evaluate --config configs/qwen3_1.7b_32k_pyramidkv.yaml --print-config

# 1) 4 条样本冒烟检查（单卡）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_pyramidkv.yaml --limit 4

# 2) 同一流程的 FullKV 对照（不删 KV，检查流水线本身）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_pyramidkv.yaml --method fullkv --limit 4

# 3) 全量（去掉 --limit），8 卡每卡一份模型
NPROC_PER_NODE=8 bash scripts/evaluate.sh configs/qwen3_1.7b_32k_pyramidkv.yaml

# 4) 预算扫描，结果分别写到 outputs/pyramidkv-b<预算>/
BUDGETS="128 512 2048" bash scripts/sweep_budget.sh configs/qwen3_1.7b_32k_pyramidkv.yaml --limit 50
```

常用覆盖项：`--max-capacity-prompt`、`--window-size`、`--kernel-size`、`--pooling`、`--beta`、`--kv-cache-granularity`、`--tasks`、`--limit`、`--output`、`--model`（SemanticLogKV 的 `ckpt_dir` 或 HF 目录）。首次在没有离线缓存的机器上运行可加 `--hf-online --hf-cache-base ~/.cache/huggingface`。

### 与 SemanticLogKV 横向对比

1. 在 SemanticLogKV 仓库按其文档生成同一组样本的结果（单档 32K，`pretrained` 用同一个 Base 路径）：

   ```bash
   DIAG_ARGS='--limit 4 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
     bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_2,niah_single_3 none
   ```

2. （可选）把 PyramidKV 的预算对齐到该结果在生成请求结束时的平均保留槽位数（按条目数对齐，不是字节）：

   ```bash
   bash scripts/evaluate.sh configs/qwen3_1.7b_32k_pyramidkv.yaml --limit 4 \
     --budget-from-result /path/to/semanticlogkv/eval_results_TIMESTAMP.json --output outputs/pyramidkv-matched
   ```

3. 核对两边的实际测试输入完全一致（逐样本 doc/prompt/target 哈希与生成参数），并并排输出分数、KV 保留率、速度和显存：

   ```bash
   python -m pyramidkv.compare_results /path/to/semanticlogkv/eval_results_TIMESTAMP.json \
     outputs/pyramidkv-matched/eval_results_TIMESTAMP.json outputs/fullkv-full/eval_results_TIMESTAMP.json
   ```

   输入不一致时直接报错（例如 `pretrained` 路径、样本数或生成参数不同）。H2O 分支的结果文件也可以一起传入。

`--method fullkv` 与 SemanticLogKV 的 `log_kv_dense_mode: true` 应给出几乎相同的 NIAH 分数；两者使用不同的 attention 实现，逐 token logits 可能有 bf16 级别差异。

## 输出

`--output` 目录（默认 `outputs/<method>-b<预算>`）中：

| 文件 | 内容 |
|---|---|
| `resolved_eval.yaml` | 解析后的全部设置（方法参数、任务、窗口、缓存目录） |
| `eval_results_<ts>.json` | 与 SemanticLogKV 相同的主字段，另含 `model_backend`、`method_config`、`provenance`（权重/tokenizer/代码 SHA-256、git 提交、依赖版本、GPU） |
| `eval_results_<ts>.kv_compression.csv` | 每次模型调用的 KV 槽位快照 |
| `eval_results_<ts>.inference_metrics.xlsx` | 速度汇总、逐请求明细、按 prompt 长度分桶（需 pandas/openpyxl） |
| `eval_results_<ts>.csv` | 各任务的全部数值指标（NIAH 的指标名为长度，如 `32768,none`） |

## 测试

CPU 即可运行，使用随机小模型，不下载任何权重或数据：

```bash
python -m pytest -q
```

覆盖：Qwen3 与 Transformers 参考实现数值一致；LitGPT/HF 两种权重布局加载一致；带 cache 的分段 prefill/续写/逐 token decode 与整段前向一致；金字塔预算公式与 PyramidKV 选择结果与 KVCache-Factory 原始代码逐元素一致；真实删除后的条目计数；请求截断、BOS、EOS、rolling 窗口、按 rank 分配；以及一次经 lm-eval 的端到端运行（生成 + 选择题，PyramidKV 与 FullKV 结果通过输入一致性校验）。

## 注意事项

- 这是免训练方法，直接在原始 Base 上评测；SemanticLogKV 使用 CPT 后的权重，比较时需注明。
- PyramidKV 的 prefill 本身是完整注意力，峰值显存包含整段 prompt 的单层 K/V 与打分工作区；它节省的是 prefill 之后常驻的 cache 和 decode 访存。
- 槽位比例不等于显存比例；`query_head` 布局下保留率可能大于 1（每个 KV group 存多份）。
- 尚未在真实 1.7B 权重和 CUDA 上运行；请先用 `--limit 4` 确认端到端链路，再扩大样本。
