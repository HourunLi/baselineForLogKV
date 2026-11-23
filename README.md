# Log-Linear Attention × niahLogKV

本分支用作者发布的 **HAttention Mamba-2** 权重（825,465,312 个参数）和原配 tokenizer，接入 `niahLogKV` 的 lm-eval 流程。模型保留因果深度卷积、逐头选择性衰减、base-2 weak 层级矩阵状态、按输入计算的层级读出、门控 RMSNorm 和 SwiGLU。作者发布的 Gated DeltaNet 是另一套模型，本分支选用 Mamba-2 版本。权重、架构和 tokenizer 均与 Qwen3-1.7B-Base 不同。[来源及提交](lla_bench/SOURCES.md)

## 启动

在仓库根目录使用服务器的 `torch218`。评测协议使用 `lm-eval==0.4.11` 和 `datasets==4.8.4`。模型通过 PyTorch 执行，无需安装 Flame、FLA、causal-conv1d 或 mamba-ssm。

```bash
source scripts/hattention_env.sh
export LLA_CHECKPOINT_DIR="$PWD/checkpoints/hattention-mamba2"
export LLA_OUTPUT_DIR="$PWD/results/hattention"

# 在能联网的节点执行一次。约 3.30 GB，下载后按固定 revision 校验所有文件。
python -m lla_bench.download "$LLA_CHECKPOINT_DIR"

# 检查实际任务、长度和分阶段去重后的清单
bash eval.sh --print-config

# 主线与 NIAH 各做少量样本，使用同一个原生模型
LIMIT=1 EVAL_LENGTHS=none bash eval.sh configs/hattention.yaml boolq niah_single_1

# 全量主线 + 32K、64K、128K、256K、512K、1M 六档长测
bash majob.sh

# 只跑两档指定长测，每个任务取两条
LIMIT=2 EVAL_MAIN=0 EVAL_LENGTHS=512k,1m EVAL_SUITES=ruler,turns bash eval.sh
```

每个节点启动一次 `majob.sh`。启动器使用平台的节点、rank、主机地址和可见 GPU 数，按 main/NIAH 两阶段运行。评测离线；原生模型和 tokenizer 从同一个检查点目录读取，输出目录可单独指定。任务显式点名时缺少数据会报错。默认清单中的可选任务沿用参考分支规则并记录跳过原因。完整七类数据都已经准备好时，可显式设置 `EVAL_SUITES=ruler,babilong,ppl,books,turns,longbench2,infinitebench` 来要求所有适用套件运行。

## 数据

默认 `HF_HOME` 为 `/workspace/data/hf_cache`。启动器按参考实现设置 hub、datasets、RULER、NLTK、BABILong 和 LongBench v2 路径及 CUDA/NCCL 环境。下载程序按 `niahLogKV/unused` 的方式使用普通文件、requests、429 重试并支持无可用 CA 的代理。模型下载验证 Git blob 或 LFS 摘要，数据准备使用参考代码的完整性检查。可用 `--verify-ssl` 开启证书验证。

```bash
source scripts/hattention_env.sh
python -m lla_bench.unused.fetch_long_eval_data \
  --lengths 32k,64k,128k,256k,512k,1m --longbench2 --infinitebench \
  --ppl-hub emozilla/pg19-test --ppl-books-hub emozilla/pg19 \
  --tokenizer "$LLA_CHECKPOINT_DIR"
```

如 LogKV 已有同一批原始数据，直接复用其路径。下载示例中的 PG19 来源应与对照实验的语料选择一致；如已有独立留出语料，使用 `--ppl-data` 和 `--ppl-books-data`。512K、1M 共用至少 1,048,576 token 的文档池，按两侧实际合格数量报告。参照实验设置了 `LONG_PPL_DOCS` 时，两边取相同值。根据下载摘要导出 `LONG_PPL_DATA` 和 `LONG_PPL_BOOKS`，让两个模型读取相同的语料来源。依赖 tokenizer 的长度筛选、PPL 文档选择与 RULER 提示生成使用各模型自己的 tokenizer。比较时保留实际文档、输入长度和截断记录。每 token PPL 和 tokens/s 都受分词粒度影响；比较 bits/byte 时也要核对被打分的文本后缀。

```bash
# 已导出共享数据路径时，在评测环境检查六档原始文件和可用文档
python -m lla_bench.unused.check_long_eval_data \
  --lengths 32k,64k,128k,256k,512k,1m --tokenizer "$LLA_CHECKPOINT_DIR"

# 多节点可提前在 CPU 上生成 512K/1M 的 RULER 数据
python -m lla_bench.unused.prepare_ruler_datasets --lengths 512k,1m \
  --tokenizer "$LLA_CHECKPOINT_DIR" --samples 100
```

## 评测口径

32K 主线沿用固定参考版本的通用评测、LongBench v1/E、32K 多轮问答与书籍 PPL，并保留多种单针、多针、词值针与提示变体。长测按参考计划保留每档 1,024 个生成位置、默认每任务 100 条、已测任务去重、停止字符串、选项对数似然和评分聚合。

| 长测套件 | 适用长度 |
|---|---|
| RULER | 六档。SQuAD 子任务到 128K。512K/1M 使用参考分支的扩展生成协议 |
| BABILong、LongBench v2、多轮 NIAH、长文本 PPL、书籍 PPL 曲线 | 六档 |
| InfiniteBench | 32K、64K、128K、256K |

作者代码的训练配置为 **16,384 token、15 级读出**。超出此范围时继续执行二进制进位，每个新增层级使用当前 token 对最深已训练层的读出权重。结果 JSON 记录 `level_extrapolation=share_deepest_trained_lambda`。SSM 没有旋转位置编码，YaRN 只适用于对照侧。六档目标窗口均超过作者的训练配置。每个请求按实际进入模型的 token 数记录 `extrapolated_read_levels`，运行汇总给出实际越过训练层级的请求数。

JSON、评分 CSV、压缩明细 CSV 与速度 XLSX 沿用参考导出。速度包含 TTFT、prefill/decode 吞吐、每 token 解码耗时、生成及打分吞吐。CUDA 峰值同时记录 allocated/reserved，使用相同的请求同步及重置边界。PyTorch 实现的数值顺序和速度与作者的融合 CUDA/Triton kernel 有差异。

每个结果另有 `*.summary.csv`。它逐行保留 lm-eval 返回的任务、组、指标、filter 和标准误，包括 RULER 各长度、逐轮准确率、PPL、bits/byte 及 LongBench 各任务的评分。旧版 `*.csv` 保留主线使用的两列格式。速度 XLSX 依实际 prompt 长度分桶，覆盖 32K 至 1M 和超出 1M 的生成边界。单条日志包含样本编号，可关联打分、截断和缓存明细。

同一命令可以从两侧的完整结果重新导出这些指标。参数使用带时间戳的结果 JSON，每次读取一个文件。相同窗口和阶段的结果应对照其 `inputs/text_set_sha256`，该摘要取自 lm-eval 的文档编号、filter 和文档、提示、答案哈希。摘要不同的两组结果可作任务级比较，逐样本比较需要相同输入；分词和截断还需结合实际长度明细。

```bash
python -m lla_bench.report /path/to/logkv/eval_results_TIMESTAMP.json \
  /path/to/hattention/eval_results_TIMESTAMP.json > comparison.csv
```

需要对照完全相同的 RULER 原始文本时，可以把 `LLA_NIAH_DATA_TOKENIZER_DIR` 设为 LogKV 实际使用的本地 Qwen tokenizer 目录。这个选项只改变独立 NIAH 阶段的样本构造。模型前向仍用作者的 tokenizer，结果的 `metadata.pretrained` 和 `tokenizer_dir` 分别记录两个路径。两边用同一套语料、协议、长度、随机种子和样本数，并核对文本摘要。按 Qwen 分词构造的题目经原生 tokenizer 编码后可能触发截断。PPL 和 main 阶段的多轮任务维持默认协议。

## 同预算比较

矩阵状态没有逐 token KV 条目，对应的 `kv_retention_ratio`、`kv_saving_ratio`、`kv_compression_factor` 均为 null。每次请求结束时记录真实转入模型的 token 数、占用的矩阵数及卷积历史字节。分配预算在 `cache_budget.persistent_cache_bytes`。另存 `live_state_byte_ratio`、节省比例和压缩倍数，分母为相同 token 数下 Qwen3-1.7B 的 dense BF16 K/V（28 层、8 个 KV 组、维度 128）。汇总采用字节总和之比。实现先读后进位，已占用矩阵数会在二次幂边界下降。整卡峰值另受模型权重、激活、工作区和 allocator 影响。

两侧都能直接计算 `common_byte_budget.allocated_byte_ratio`。令 B 为每请求复用的持久缓存字节数，N 为请求数，T 为各请求实际进入模型的 token 数之和。该比值为 `N × B / (T × 114688)`，分母按 batch 1、Qwen3-1.7B dense BF16 KV 计。短请求也持有预分配缓冲，比例可以超过 1。与训练框架、tokenizer 无关的绝对比较使用同一配置窗口的 `persistent_cache_bytes`。

下表以 batch 1、BF16 模型和卷积历史、FP32 矩阵状态计。数值来自 CPU 上的持久缓冲尺寸，单位 MiB。默认参考为固定提交的 `eval_full_word_first.yaml`。最后两列给出**一组跨六档固定**的 LogKV 参数，便于另跑同预算对照。保留 P=1024、K=12，设置 `log_kv_B=26`、`log_kv_recent_size=512`、`log_kv_prefill_block=512`、`log_kv_semantic_flush_granularity=512`。该配置需要在 LogKV 侧重新评测，旧质量分对应原预算。

| 长测 | HAttention | 默认 LogKV | 固定同预算 LogKV | HAttention 相对同预算差 |
|---|---:|---:|---:|---:|
| 32K | 504.53 | 1937.75 | 494.66 | +2.00% |
| 64K | 536.03 | 2113.54 | 531.52 | +0.85% |
| 128K | 567.53 | 2292.16 | 554.21 | +2.40% |
| 256K | 599.03 | 2473.62 | 595.32 | +0.62% |
| 512K | 630.53 | 2657.91 | 639.27 | -1.37% |
| 1M | 662.03 | 2845.04 | 664.79 | -0.42% |

```bash
# 重算全部窗口，并可查看各档独立最接近的 B/recent 候选
python -m lla_bench.budget --model-config "$LLA_CHECKPOINT_DIR/config.json" --match-logkv

# 也可读取当前实验实际产出的 LogKV JSON，指定该次运行的模型窗口
python -m lla_bench.budget --model-config "$LLA_CHECKPOINT_DIR/config.json" \
  --window 1049600 --reference-json /path/to/logkv/eval_results.json
```

同预算复评配置在 `configs/logkv_word_first_matched.yaml`。将此文件复制到 **LogKV 的 niahLogKV 分支** `exp/qwen1.7b-32k/`，与 `eval_full_word_first.yaml` 同目录。在 LogKV 仓库执行下面的命令，沿用该实验的 checkpoint、任务和数据，为新预算单独出分。

```bash
bash eval.sh exp/qwen1.7b-32k/logkv_word_first_matched.yaml
```

LogKV 的动态有效槽位还取决于输入和路由。上表匹配持久分配，实际峰值及任务质量应直接读取同硬件、同任务两侧的输出。

## 验证

```bash
python -m pytest -q tests/test_native_hattention.py tests/test_hattention_reporting.py
```

- 在 Python 3.10.20、PyTorch 2.11.0 CPU、lm-eval 0.4.11、datasets 4.8.4、transformers 4.51.3 上，五项回归测试通过。使用作者完整权重、原配 tokenizer，经 `LLA_EVAL_DEVICE=cpu` 和 `eval.sh` 完成 BoolQ 单样本的两个选项打分。
- 使用完整 FP32 权重，流式前向与作者 `make_masked_H_matrix` 的逐层矩阵实现比较，8 个位置共 256,000 个 logits，最大绝对误差为 1.14e-5。此核对在 CPU 上执行。
- 六档任务名、窗口、分阶段任务清单与固定参照提交逐档核对。使用小尺寸随机模型跑通 lm-eval 的真实 1M RULER 生成和 PG19 评分，前向分别消费 1,048,065 和 1,048,577 个 token，并导出长度分数、PPL、bits/byte、资源汇总和 XLSX。另用可精确求和的递推输入核对全部 1,049,600 个位置。这些测试验证长窗口和指标管线。
- A800 上的整卡峰值、速度以及公开权重的全量质量分需要在目标环境实测。请保留双方的检查点、数据、长度、样本数和实际执行设置。
