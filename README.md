# H2O baseline（Qwen3-1.7B-Base，SemanticLogKV 评测口径）

本分支是一个**独立项目**：在 Qwen3 上实现 H2O（Heavy-Hitter Oracle），并按 SemanticLogKV 的数据集、请求接口和指标口径评测，用于横向对比。

- H2O 的 prompt 压缩规则移植自 [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)（MIT），见 [LICENSE](LICENSE)；可选的 decode 期淘汰按 [H2O 原实现](https://github.com/FMInference/H2O) 的 real-drop 策略自行实现。
- 评测侧的模型、tokenizer、KV cache、生成、打分、指标和结果格式全部在本仓库实现。**不 import、不加载 litgpt / SemanticLogKV 的任何代码**，只依赖 PyTorch、tokenizers、lm-eval 等通用库。
- 与 SemanticLogKV 对齐的内容是“行为”：同一份原始 Base 权重、同一 tokenizer 规则、同一 lm-eval 任务与缓存、同样的截断/生成/打分规则、同样的 KV 槽位与速度指标定义、同样的结果文件布局。

## KVCache-Factory 是怎么比较不同方法的

KVCache-Factory 把多种 KV 压缩方法放在同一个推理入口下，用参数切换：

| 组件 | 作用 |
|---|---|
| `pyramidkv/monkeypatch.py` 的 `replace_llama(method)` / `replace_mistral(method)` | 按 `--method` 把 HF `LlamaAttention.forward` 等替换成对应方法（`h2o`、`pyramidkv`、`snapkv`、`streamingllm`、`fullkv` 等） |
| `pyramidkv/pyramidkv_utils.py` 的 `*KVCluster.update_kv` | 每种方法的核心：prefill 时用完整 KV 计算注意力，然后只把选中的 KV 写进 cache；decode 时在压缩后的 cache 上追加 |
| `run_longbench.py` / `run_needle_in_haystack.py` / `run_ruler.py` | 三个评测 runner；`--max_capacity_prompts` 设定每层预算，`--kv_cache_granularity`、`--gqa_score_agg` 控制 GQA 布局 |
| `scripts/scripts_*/eval.sh`、`eval_methods.sh`、`metrics.sh` | 单次运行、预算扫描（64…4096）与打分汇总 |

例如 `python run_longbench.py --method h2o --model_path ... --max_capacity_prompts 128 --attn_implementation flash_attention_2`；换成 `--method pyramidkv` 即比较 PyramidKV。

不能直接拿它和 SemanticLogKV 比：它只 patch Llama/Mistral（锁定 transformers 4.44.2，没有 Qwen3 的 q/k norm），其 RULER/LongBench 数据、prompt 模板、HF `generate` 停止规则和打分都与 SemanticLogKV 不同；它的 H2O 一次性构造 L×L 注意力矩阵，32K prompt 需要几十 GB。因此本分支只移植 H2O 的算法核心，再按 SemanticLogKV 的接口自己实现评测。

## 方法实现

每层的流程：

1. **Prefill**：用整段 prompt 的完整 K/V 做因果注意力（prefill 输出不受压缩影响）。
2. **累计注意力评分**：每个旧 key 的分数 = 所有 prompt query 对它的注意力概率之和（因果 softmax），GQA 下按组聚合。按 query 分块计算，不生成完整 L×L 矩阵；累加用 fp32。
3. **淘汰**：保留分数最高的 `max_capacity_prompt - window_size` 个旧 token（heavy hitters）和最后 `window_size` 个 token，其余 K/V 不再存储。prompt 短于预算时不压缩。各层预算相同。
4. **Decode / 续写**：默认（KVCache-Factory 语义）新 token 直接追加，不再淘汰。
5. **`--decode-evict`（H2O 原论文语义）**：cache 保留每个条目的累计分数；之后每次前向显式计算注意力，把新的注意力概率累加到分数上，超出预算时淘汰最近窗口以外分数最低的条目，cache 始终保持 `max_capacity_prompt`。

| 参数 | 默认值 | 来源/说明 |
|---|---|---|
| `max_capacity_prompt` | 2048 | KVCache-Factory 初始化默认 |
| `window_size` | 8 | KVCache-Factory runner 对 `h2o` 的设置；H2O 原论文通常 heavy 与 recent 各占一半，可设为预算的一半 |
| `decode_evict` | false | true 时 decode 期持续淘汰（原 H2O） |
| `score_chunk_size` | 512 | 评分分块大小，只影响显存和速度，结果相同 |
| `kv_cache_granularity` | `kv_head` | 每个 KV head 一份条目，与 SemanticLogKV 的 KV group 槽位同单位；`query_head` 复现旧布局（GQA 下条目翻倍） |
| `gqa_score_agg` | `mean` | 组内 query head 分数聚合 |

分块评分在测试中与完整注意力矩阵逐元素比较；保留结果与 KVCache-Factory 原始 `H2OKVCluster` 代码逐元素一致。Qwen3 模型本身与 Transformers 的 `Qwen3ForCausalLM` 做了数值对照。

**选择题/PPL 打分**（`--loglikelihood-mode`）：H2O 只在一次前向结束后压缩，同一次前向内的 token 看到的都是完整 KV。默认 `compress_context` 先 prefill 并压缩 context，再在压缩后的 cache 上为 continuation 打分（与生成答案时的处境一致）；`single_forward` 一次前向打完分，结果等同 FullKV。rolling PPL 的 context 只有 1 个 token，不会触发压缩。NIAH 是生成任务，不受此选项影响。

## 与 SemanticLogKV 的对齐

| 项目 | 本分支做法 |
|---|---|
| 权重 | 直接读取 SemanticLogKV 实验 `ckpt_dir` 下的原始 `lit_model.pth`（参数名按 LitGPT 布局，无需转换），也支持 HF safetensors；严格加载，缺/多张量即报错 |
| tokenizer | 自行实现与参考评测相同的规则：BOS/EOS 来自 `tokenizer_config.json` 与 `generation_config.json`，自动去除后端多加的 BOS/EOS，不加 chat template |
| 任务与数据 | lm-eval 0.4.9；默认 `niah_single_2,niah_single_3`；`metadata.max_seq_lengths=[32768]`，`metadata.pretrained` 原样使用 Base 路径字符串；HF/RULER 离线缓存根目录相同 |
| 截断 | 打分时左截断 context；生成时先预留 `max_gen_toks` 再左截断 prompt；空 context 用 `[bos]` |
| 生成 | 非采样时 temperature/top_p 置 0 走 argmax；只在 EOS 或长度上限停止，不执行 `until` 字符串（与参考评测一致） |
| 打分 | continuation `bos=False` 单独编码；fp32 log-softmax，1024 位置分块；rolling 窗口 `max_seq_length-1` |
| KV 指标 | 请求结束、释放 cache 前快照；分母为已前向 token × 层 × KV group，不含最后一个未回填的输出 token；汇总用 ΣS/ΣD；`recent`=最近窗口（含追加 token），`exact`=保留的 heavy hitters，`hierarchical`=0 |
| 速度/显存 | 同样的 prefill（含首 token 采样，H2O 评分与淘汰计入 prefill）/decode 计时边界与每请求 CUDA 峰值 |
| 多卡 | 每卡一份模型，按 `requests[rank::world]` 分配，Gloo 汇总后按原顺序重排 |
| 输出 | `eval_results_<时间>.json`（`cache_budget`、`kv_compression`、`inference_metrics`、`metadata`、`results`，含 lm-eval 逐样本日志）及 `.kv_compression.csv`、`.inference_metrics.xlsx`、`.csv`、`resolved_eval.yaml` |

`cache_budget.persistent_cache_bytes` 为 `null`：cache 按需增长，没有 SemanticLogKV 那样的预分配常驻缓冲；逐请求的 `retained_kv_bytes` 给出保留条目的 K/V 字节数（不含预留空位与临时打分工作区）。

## 安装

建议与 SemanticLogKV 评测使用同一个 CUDA 环境（相同的 torch、lm-eval、datasets 版本）：

```bash
git clone -b H2O https://github.com/HourunLi/baselineForLogKV.git H2O
cd H2O
pip install -e '.[eval,test]'
```

lm-eval 0.4.9 与 transformers 5.x 不兼容，`[eval]` 已限制 `transformers<5`。prefill 使用 PyTorch SDPA，不依赖 FlashAttention 包。

## 运行方式

以下命令都在仓库根目录执行。

```bash
# 0) 只解析配置，不加载权重和数据
python -m h2okv.evaluate --config configs/qwen3_1.7b_32k_h2o.yaml --print-config

# 1) 4 条样本冒烟检查（单卡）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --limit 4

# 2) H2O 原论文语义：decode 期持续淘汰，heavy/recent 各占一半
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --limit 4 --decode-evict --window-size 1024

# 3) 同一流程的 FullKV 对照（不删 KV，检查流水线本身）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --method fullkv --limit 4

# 4) 全量（去掉 --limit），8 卡每卡一份模型
NPROC_PER_NODE=8 bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml

# 5) 预算扫描，结果分别写到 outputs/h2o-b<预算>/
BUDGETS="128 512 2048" bash scripts/sweep_budget.sh configs/qwen3_1.7b_32k_h2o.yaml --limit 50
```

常用覆盖项：`--max-capacity-prompt`、`--window-size`、`--decode-evict`、`--score-chunk-size`（显存紧张时调小）、`--kv-cache-granularity`、`--tasks`、`--limit`、`--output`、`--model`（SemanticLogKV 的 `ckpt_dir` 或 HF 目录）。首次在没有离线缓存的机器上运行可加 `--hf-online --hf-cache-base ~/.cache/huggingface`。

### 与 SemanticLogKV 横向对比

1. 在 SemanticLogKV 仓库按其文档生成同一组样本的结果（单档 32K，`pretrained` 用同一个 Base 路径）：

   ```bash
   DIAG_ARGS='--limit 4 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
     bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_2,niah_single_3 none
   ```

2. （可选）把 H2O 的预算对齐到该结果在生成请求结束时的平均保留槽位数（按条目数对齐，不是字节）：

   ```bash
   bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --limit 4 \
     --budget-from-result /path/to/semanticlogkv/eval_results_TIMESTAMP.json --output outputs/h2o-matched
   ```

3. 核对实际测试输入完全一致（逐样本 doc/prompt/target 哈希与生成参数），并并排输出分数、KV 保留率、速度和显存：

   ```bash
   python -m h2okv.compare_results /path/to/semanticlogkv/eval_results_TIMESTAMP.json \
     outputs/h2o-matched/eval_results_TIMESTAMP.json outputs/fullkv-full/eval_results_TIMESTAMP.json
   ```

   输入不一致时直接报错（例如 `pretrained` 路径、样本数或生成参数不同）。PyramidKV 分支的结果文件也可以一起传入。

`--method fullkv` 与 SemanticLogKV 的 `log_kv_dense_mode: true` 应给出几乎相同的 NIAH 分数；两者使用不同的 attention 实现，逐 token logits 可能有 bf16 级别差异。

## 输出

`--output` 目录（默认 `outputs/<method>-b<预算>`，decode 淘汰时追加 `-evict`）中：

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

覆盖：Qwen3 与 Transformers 参考实现数值一致；LitGPT/HF 两种权重布局加载一致；带 cache 的分段 prefill/续写/逐 token decode 与整段前向一致；分块累计注意力与完整矩阵一致；H2O 保留结果与 KVCache-Factory 原始代码逐元素一致；decode 淘汰保持预算且淘汰最近窗口外分数最低的条目；请求截断、BOS、EOS、rolling 窗口、按 rank 分配；以及一次经 lm-eval 的端到端运行（生成 + 选择题，H2O、H2O+decode 淘汰与 FullKV 结果通过输入一致性校验）。

## 注意事项

- 这是免训练方法，直接在原始 Base 上评测；SemanticLogKV 使用 CPT 后的权重，比较时需注明。
- H2O 的评分需要对整段 prompt 再算一遍注意力（分块），32K 时 prefill 明显慢于 FullKV；峰值显存包含单层完整 K/V 与 `score_chunk_size × L` 的打分工作区。
- `--decode-evict` 下 decode 用显式注意力（而非 SDPA）以获得注意力概率，decode 速度不能直接与默认模式对比。
- 槽位比例不等于显存比例；`query_head` 布局下保留率可能大于 1（每个 KV group 存多份）。
- 尚未在真实 1.7B 权重和 CUDA 上运行；请先用 `--limit 4` 确认端到端链路，再扩大样本。
