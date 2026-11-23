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

# 下载期间每 5 秒打印已下载 MiB、百分比、最近速度及预计剩余秒数。最后一行显示全文件平均速度。

# 检查实际任务、长度和分阶段去重后的清单
bash eval.sh --print-config

# 主线与 NIAH 各做少量样本，使用同一个原生模型
LIMIT=1 EVAL_LENGTHS=none bash eval.sh configs/hattention.yaml boolq niah_single_1

# 全量主线 + 32K、64K、128K、256K、512K、1M 六档长测
bash majob.sh

# 只跑两档指定长测，每个任务取两条
LIMIT=2 EVAL_MAIN=0 EVAL_LENGTHS=512k,1m EVAL_SUITES=ruler,turns bash eval.sh
```

每个节点启动一次 `majob.sh`。启动器使用平台的节点、rank、主机地址和可见 GPU 数，按 main/NIAH 两阶段运行。评测离线；检查点、tokenizer 和输出目录可以独立指定。任务显式点名时缺少数据会报错。默认清单中的可选任务沿用参考分支规则并记录跳过原因。完整七类数据都已经准备好时，可显式设置 `EVAL_SUITES=ruler,babilong,ppl,books,turns,longbench2,infinitebench` 来要求所有适用套件运行。

## 数据

默认 `HF_HOME` 为 `/workspace/data/hf_cache`。启动器按参考实现设置 hub、datasets、RULER、NLTK、BABILong 和 LongBench v2 路径及 CUDA/NCCL 环境。下载程序按 `niahLogKV/unused` 的方式使用普通文件、requests、429 重试并支持无可用 CA 的代理。模型下载验证 Git blob 或 LFS 摘要，数据准备使用参考代码的完整性检查。可用 `--verify-ssl` 开启证书验证。

```bash
source scripts/hattention_env.sh
python -m lla_bench.unused.fetch_long_eval_data \
  --lengths 32k,64k,128k,256k,512k,1m --longbench2 --infinitebench \
  --ppl-hub emozilla/pg19-test --ppl-books-hub emozilla/pg19 \
  --tokenizer "$LLA_CHECKPOINT_DIR"
```

如 LogKV 已有同一批原始数据，直接复用其路径。根据下载摘要导出 `LONG_PPL_DATA` 和 `LONG_PPL_BOOKS`，让两个模型读取相同的语料来源。依赖 tokenizer 的长度筛选、PPL 文档选择与 RULER 提示生成使用各模型自己的 tokenizer。比较时保留实际文档、输入长度和截断记录。每 token PPL 和 tokens/s 都受分词粒度影响。

## 评测口径

32K 主线沿用固定参考版本的通用评测、LongBench v1/E、32K 多轮问答与书籍 PPL，并保留多种单针、多针、词值针与提示变体。长测按参考计划保留每档 1,024 个生成位置、默认每任务 100 条、已测任务去重、停止字符串、选项对数似然和评分聚合。

| 长测套件 | 适用长度 |
|---|---|
| RULER | 六档。SQuAD 子任务到 128K。512K/1M 使用参考分支的扩展生成协议 |
| BABILong、LongBench v2、多轮 NIAH、长文本 PPL、书籍 PPL 曲线 | 六档 |
| InfiniteBench | 32K、64K、128K、256K |

作者代码的训练配置为 **16,384 token、15 级读出**。超出此范围时继续执行二进制进位，每个新增层级使用当前 token 对最深已训练层的读出权重。结果 JSON 记录 `level_extrapolation=share_deepest_trained_lambda`。SSM 没有旋转位置编码，YaRN 只适用于对照侧。全部六档都属于该 checkpoint 的训练长度外推。

JSON、评分 CSV、压缩明细 CSV 与速度 XLSX 沿用参考导出。速度包含 TTFT、prefill/decode 吞吐、每 token 解码耗时、生成及打分吞吐。CUDA 峰值同时记录 allocated/reserved，使用相同的请求同步及重置边界。PyTorch 实现的数值顺序和速度与作者的融合 CUDA/Triton kernel 有差异。

## 论文汇总表

`lla_bench.summary` 沿用参考版本的表头、指标映射和分组平均。可同时读两个模型的运行日志或带时间戳的结果 JSON，每行一个模型、共 458 列。六档 RULER、BABILong、LongBench v2、多轮问答、PPL 曲线及支持长度内的 InfiniteBench 都有独立列。仅传同一实验的文件，未测项目留空。和参考实现一样，各组平均按可用项计算，小样本或缺项结果需要在组间比较前核对覆盖范围。

```bash
# 各自将完整评测的 stdout/stderr 存为日志后
python -m lla_bench.summary --model HAttention hattention.log \
  --model LogKV logkv.log --output comparison.csv
```

## 同预算比较

矩阵状态没有逐 token KV 条目，对应的 `kv_retention_ratio`、`kv_saving_ratio`、`kv_compression_factor` 均为 null。每次请求结束时记录真实转入模型的 token 数、占用的矩阵数及卷积历史字节。分配预算在 `cache_budget.persistent_cache_bytes`。另存 `live_state_byte_ratio`、节省比例和压缩倍数，分母为相同 token 数下 Qwen3-1.7B 的 dense BF16 K/V（28 层、8 个 KV 组、维度 128）。汇总采用字节总和之比。实现先读后进位，已占用矩阵数会在二次幂边界下降。整卡峰值另受模型权重、激活、工作区和 allocator 影响。

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

LogKV 的动态有效槽位还取决于输入和路由。上表匹配持久分配，实际峰值及任务质量应直接读取同硬件、同任务两侧的输出。

## 验证

`python -m pytest -q tests/test_native_hattention.py` 检查门控因果定义、任意 prefill 分块、跨层级进位、逐 token 解码、重置与字节计数。本分支开发时还用作者公开权重、原配 tokenizer、上游 `make_masked_H_matrix` 做了独立 FP32 前向核对；7 个位置共 224,000 个 logits，最大绝对误差 7.63e-6。CPU 小型状态流覆盖 1,049,600 个位置。使用作者权重在 CPU 上逐路跑通了 BoolQ 1 题（2 个选项）以及 32K `niah_single_1` 1 题（输入 32,585 token、生成 128 token、无截断），均产生 JSON、CSV、XLSX。两个样本均得 0 分，仅用于检查端到端执行。目标 GPU 上的全套质量、速度和显存峰值仍待实测。
