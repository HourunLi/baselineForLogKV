# H2O baseline（Qwen3-1.7B-Base，SemanticLogKV 评测口径）

本分支是一个**独立项目**：在 Qwen3 上实现 H2O（Heavy-Hitter Oracle），并按 SemanticLogKV 的数据集、请求接口和指标口径评测，用于横向对比。

- H2O 的 prompt 压缩规则移植自 [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)（MIT），见 [LICENSE](LICENSE)；可选的 decode 期淘汰按 [H2O 原实现](https://github.com/FMInference/H2O) 的 real-drop 策略自行实现。
- 评测侧的模型、tokenizer、KV cache、生成、打分、指标和结果格式全部在本仓库实现。**不 import、不加载 litgpt / SemanticLogKV 的任何代码**，只依赖 PyTorch、tokenizers、lm-eval 等通用库。
- 与 SemanticLogKV 对齐的内容是“行为”：同一份原始 Base 权重、同一 tokenizer 规则、同一 lm-eval 任务与缓存、同样的截断/生成/打分规则、同样的 KV 槽位与速度指标定义、同样的结果文件布局。
- 统一实现规则见 [AGENTS.md](AGENTS.md) 与 [Baseline 实现规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md)。正式评测只用 32K 窗口，完整任务为项目内 [configs/eval_suites.yaml](configs/eval_suites.yaml) 维护的 47 个 main 入口和 3 个 NIAH（仅 `[32768]`），由 `majob.sh` 分两阶段运行。

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
| 任务与数据 | lm-eval 0.4.9；`--suite main` 为参照版本 `eval.sh` / `majob.sh` 的 47 个 main 入口（14 通用、20 LongBench、13 LongBench-E，任务组由 lm-eval 展开），不传 NIAH metadata；`--suite niah` 为 `niah_single_1/2/3`，`metadata.max_seq_lengths=[32768]`，`metadata.pretrained` 原样使用 Base 路径字符串；HF/RULER 离线缓存根目录相同，评测强制离线，不提供联网下载开关 |
| 截断 | 打分时左截断 context；生成时先预留 `max_gen_toks` 再左截断 prompt；空 context 用 `[bos]` |
| 生成 | 非采样时 temperature/top_p 置 0 走 argmax；只在 EOS 或长度上限停止，不执行 `until` 字符串（与参考评测一致） |
| 打分 | continuation `bos=False` 单独编码；fp32 log-softmax，1024 位置分块；rolling 窗口 `max_seq_length-1` |
| KV 指标 | 请求结束、释放 cache 前快照；分母为已前向 token × 层 × KV group，不含最后一个未回填的输出 token；汇总用 ΣS/ΣD；`recent`=最近窗口（含追加 token），`exact`=保留的 heavy hitters，`hierarchical`=0 |
| 速度/显存 | 同样的 prefill（含首 token 采样，H2O 评分与淘汰计入 prefill）/decode 计时边界与每请求 CUDA 峰值 |
| 多卡 | 每卡一份模型，按 `requests[rank::world]` 分配，Gloo 汇总后按原顺序重排 |
| 预处理缓存 | 原始离线 `load_dataset` 缓存照常读取；map/filter 结果用 `datasets.disable_caching()` 写进程私有临时文件，多 rank 不竞争共享挂载上的同一缓存文件 |
| 输出 | `eval_results_<run_id>.json`（`cache_budget`、`kv_compression`、`inference_metrics`、`metadata`、`results`，含 lm-eval 逐样本日志）及 `.kv_compression.csv`、`.inference_metrics.xlsx`、`.csv`、`resolved_eval_<run_id>.yaml`；`run_id` 为启动时间，同目录已存在时自动加后缀，不覆盖历史结果 |

`cache_budget.persistent_cache_bytes` 为 `null`：cache 按需增长，没有 SemanticLogKV 那样的预分配常驻缓冲；逐请求的 `retained_kv_bytes` 给出保留条目的 K/V 字节数（不含预留空位与临时打分工作区）。

## 安装

建议与 SemanticLogKV 评测使用同一个 CUDA 环境（相同的 torch、lm-eval、datasets 版本）：

```bash
git clone -b H2O https://github.com/HourunLi/baselineForLogKV.git H2O
cd H2O
pip install -e '.[eval,test]'
```

`[eval]` 安装 `lm-eval[ruler,longbench,ifeval]==0.4.9`、`datasets>=2.18,<4`；评测入口启动时检查 lm-eval 版本和 datasets 主版本，不满足即报错。lm-eval 0.4.9 与 transformers 5.x 不兼容，`[eval]` 已限制 `transformers<5`。prefill 使用 PyTorch SDPA，不依赖 FlashAttention 包。范围约束不能代替环境记录：每个结果的 `provenance` 写入实际解释器、torch/lm-eval/datasets/pyarrow/transformers 等版本、CUDA 运行时与驱动、各 rank 的主机与 GPU。

## 运行方式

以下命令都在仓库根目录执行。

每次运行必须二选一：`--suite main|niah|longbench`（项目内任务清单，输出写入 `<output>/<suite>/`）或 `--tasks`（诊断用的显式任务列表）。

```bash
# 0) 只解析配置，不加载权重和数据
python -m h2okv.evaluate --config configs/qwen3_1.7b_32k_h2o.yaml --suite main --print-config
python -m h2okv.evaluate --config configs/qwen3_1.7b_32k_h2o.yaml --suite niah --print-config

# 1) 每个任务 4 条样本的冒烟检查（单卡）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah --limit 4
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite main --limit 4

# 2) H2O 原论文语义：decode 期持续淘汰，heavy/recent 各占一半
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah --limit 4 --decode-evict --window-size 1024

# 3) 同一流程的 FullKV 对照（不删 KV，检查流水线本身）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah --method fullkv --limit 4

# 4) 预算扫描（可选，不属于默认流程），结果分别写到 outputs/h2o-b<预算>/<suite>/
BUDGETS="128 512 2048" bash scripts/sweep_budget.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah --limit 50
```

常用覆盖项：`--max-capacity-prompt`、`--window-size`、`--decode-evict`、`--score-chunk-size`（显存紧张时调小）、`--kv-cache-granularity`、`--limit`、`--output`、`--model`（SemanticLogKV 的 `ckpt_dir` 或 HF 目录）、`--hf-cache-base`（已准备好的离线缓存根目录）。评测从不联网下载数据；缺数据或缓存条目与 datasets<4 不兼容时直接报错，需要先修复缓存条目，不能为绕过错误升级 datasets 或换数据。

### 训练平台作业：majob.sh（完整评测）

`majob.sh` 是正式完整评测入口，**每个节点执行一次**：先 main（47 个入口），再 NIAH（3 个入口，仅 32768）。两个阶段各自一次 `torch.distributed.run`、各用一个端口，输出分别在 `<output>/main/` 与 `<output>/niah/`；main 失败时返回非零并且不启动 NIAH。它不允许 `--suite/--tasks/--eval-suites/--include-path/--ruler-lengths/--ruler-pretrained/--max-seq-length` 等会改变协议的参数，只接受方法、预算、输出目录与调试用 `--limit`。

```bash
cd /path/to/H2O
# 先在目标镜像中安装评测依赖：pip install -e '.[eval]'
bash majob.sh configs/qwen3_1.7b_32k_h2o.yaml --limit 4 --output outputs/h2o-32k-smoke   # 小样本检查
bash majob.sh configs/qwen3_1.7b_32k_h2o.yaml --output outputs/h2o-32k-full              # 全量
```

| 平台变量 | 用途 |
|---|---|
| `MA_NUM_GPUS` | 每节点 GPU 进程数；缺省依次使用 `NPROC_PER_NODE`、`LOCAL_WORLD_SIZE`、8 |
| `MA_NUM_HOSTS` | 节点数；缺省使用 `GROUP_WORLD_SIZE` 或 1 |
| `VC_TASK_INDEX` | 节点 rank；缺省使用 `NODE_RANK`、`GROUP_RANK` 或 0 |
| `MA_VJ_NAME`、`MA_TASK_NAME`、`MA_MASTER_INDEX` | 主节点 DNS（`<作业>-<任务>-<序号>.<作业>`）；无作业名时使用 `MASTER_ADDR`，多节点禁止 localhost |
| `MASTER_PORT` | 默认 6000；main 用 +1、NIAH 用 +2，可由 `MAIN_MASTER_PORT`、`NIAH_MASTER_PORT` 覆盖 |
| `H2O_PYTHON` / `H2O_CONDA_ENV` | 直接指定解释器；否则激活 `/home/ma-user/anaconda3/bin/activate ${H2O_CONDA_ENV:-torch218}`。环境名不代表实际版本，以结果中的 `provenance` 为准 |

脚本导出 `KV_BASELINE_EXPECTED_WORLD_SIZE=节点数×每节点 GPU`，评测入口发现实际进程数不同时报错。所有节点需读到同一路径的代码、Base、离线缓存，并写同一输出目录。

### 与 SemanticLogKV 横向对比

1. 在 SemanticLogKV 自己的项目里生成同一组样本的结果（以 NIAH 为例：单档 32K，`pretrained` 用同一个 Base 路径；main 阶段按其默认任务列表、不传 metadata）。本项目不读取、不调用 SemanticLogKV 的代码或配置，下面的命令在对方项目中执行：

   ```bash
   DIAG_ARGS='--limit 4 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
     bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_1,niah_single_2,niah_single_3 none
   ```

2. （可选）把 H2O 的预算对齐到该结果在生成请求结束时的平均保留槽位数（按条目数对齐，不是字节）：

   ```bash
   bash scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah --limit 4 \
     --budget-from-result /path/to/semanticlogkv/eval_results_TIMESTAMP.json --output outputs/h2o-matched
   ```

3. 核对评测协议与实际测试输入一致，并并排输出分数、KV 保留率、速度和显存：

   ```bash
   python -m h2okv.compare_results /path/to/semanticlogkv/eval_results_TIMESTAMP.json \
     outputs/h2o-matched/niah/eval_results_RUNID.json outputs/fullkv-full/niah/eval_results_RUNID.json
   ```

   协议检查分两层：逐样本 doc/prompt/target 哈希与请求参数（含生成参数）必须一致；lm-eval 记录的 `limit`、各任务样本数、任务版本与任务设置（输出类型、fewshot、生成参数、分隔符）必须一致；本项目结果额外记录的窗口长度、tokenizer 哈希和 lm-eval/datasets/pyarrow 版本若双方都有则必须一致，SemanticLogKV 自身结果缺这些字段时列为“未核对”。任一不一致直接报错。方法字段（后端、预算、权重哈希）单独列出，不参与判定。PyramidKV 分支的结果文件也可以一起传入。

`--method fullkv` 与 SemanticLogKV 的 `log_kv_dense_mode: true` 应给出几乎相同的 NIAH 分数；两者使用不同的 attention 实现，逐 token logits 可能有 bf16 级别差异。

## 输出

`--output` 目录（默认 `outputs/<method>-b<预算>`，decode 淘汰时追加 `-evict`；使用 `--suite` 时再进入 `<suite>/` 子目录）中：

| 文件 | 内容 |
|---|---|
| `resolved_eval_<run_id>.yaml` | 本次运行解析后的全部设置（方法参数、任务、窗口、缓存目录） |
| `eval_results_<run_id>.json` | 与 SemanticLogKV 相同的主字段，另含 `run_id`、`suite`、`model_backend`、`method_config`、`provenance`（权重/tokenizer/代码 SHA-256、git 提交、依赖版本、解释器、CUDA/驱动、各 rank 主机与 GPU） |
| `eval_results_<run_id>.kv_compression.csv` | 每次模型调用的 KV 槽位快照 |
| `eval_results_<run_id>.inference_metrics.xlsx` | 速度汇总、逐请求明细、按 prompt 长度分桶（需 pandas/openpyxl） |
| `eval_results_<run_id>.csv` | 各任务的全部数值指标（NIAH 的指标名为长度，如 `32768,none`） |

KV 指标分三类分别报告：逻辑 KV 条目（`kv_compression`，ΣS/ΣD）、保留条目的 K/V 字节（逐请求 `retained_kv_bytes`，不含预留空位与打分工作区）、实际 CUDA 峰值显存（`inference_metrics` 的 `peak_memory_*`）。三者不能互相替代。

## 测试

CPU 即可运行，使用随机小模型，不下载任何权重或数据：

```bash
python -m pytest -q
```

覆盖：Qwen3 与 Transformers 参考实现数值一致；LitGPT/HF 两种权重布局加载一致；带 cache 的分段 prefill/续写/逐 token decode 与整段前向一致；分块累计注意力与完整矩阵一致；H2O 保留结果与 KVCache-Factory 原始代码逐元素一致；decode 淘汰保持预算且淘汰最近窗口外分数最低的条目；请求截断、BOS、EOS、rolling 窗口、按 rank 分配；任务清单（47 main、33 LongBench、3 NIAH 仅 32768）、main 阶段不传 NIAH metadata、`majob.sh` 的节点/端口/world size 解析与首阶段失败即停止、协议字段不一致时拒绝比较、运行编号不覆盖历史结果；以及一次经 lm-eval 的端到端运行（生成 + 选择题，H2O、H2O+decode 淘汰与 FullKV 结果通过输入一致性校验）。验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 注意事项

- 这是免训练方法，直接在原始 Base 上评测；SemanticLogKV 使用 CPT 后的权重，比较时需注明。
- H2O 的评分需要对整段 prompt 再算一遍注意力（分块），32K 时 prefill 明显慢于 FullKV；峰值显存包含单层完整 K/V 与 `score_chunk_size × L` 的打分工作区。
- `--decode-evict` 下 decode 用显式注意力（而非 SDPA）以获得注意力概率，decode 速度不能直接与默认模式对比。
- 槽位比例不等于显存比例；`query_head` 布局下保留率可能大于 1（每个 KV group 存多份）。
- 尚未在真实 1.7B 权重、真实离线缓存和 CUDA/多节点作业上运行，也没有全量结果；请先用 `majob.sh ... --limit 4` 确认端到端链路，再去掉 `--limit` 做全量。
