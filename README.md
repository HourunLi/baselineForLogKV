# Log-Linear Attention × niahLogKV

默认配置使用作者发布的 **HAttention Mamba-2** 权重（825,465,312 个参数）和原配 tokenizer，接入 `niahLogKV` 的 lm-eval 流程。模型保留因果深度卷积、逐头选择性衰减、base-2 weak 层级矩阵状态、按输入计算的层级读出、门控 RMSNorm 和 SwiGLU。作者发布的 Gated DeltaNet 是另一套模型，本分支选用 Mamba-2 版本。权重、架构和 tokenizer 均与 Qwen3-1.7B-Base 不同。[来源及提交](lla_bench/SOURCES.md)

## Qwen3-8B 转换、蒸馏与 32K CPT

这一入口复用 Qwen3-8B-Base 的 tokenizer、embedding、36 层 RMSNorm 和 SwiGLU、最终 RMSNorm 与 LM head。每层替换一个原生 Mamba-2 HAttention mixer，宽度 4096、32 头、8 组、状态及值维度均为 128。转换后模型有 8,876,197,248 个参数，其中 mixer 有 2,195,420,544 个。新 mixer 独立初始化，输出投影从零开始。32K 有 16 个可训练读出级别，超长时复用当前 token 对最深已训练级别的读出权重。

managed_cluster 默认使用已有的 `/opt/conda/envs/torch218/bin/python`，启动时打印实际解释器。共享环境直接运行仓库代码，无需安装项目或训练 extra。独立管理的训练环境可用 `LLA_TRAIN_PYTHON` 指定。参照栈为 PyTorch 2.11.0+cu128、Transformers 4.51.3 和 LitData 0.2.61。教师走 PyTorch SDPA，学生使用项目内的 PyTorch HAttention，多卡由 torchrun、NCCL 和 FSDP 分片。[服务器环境清单](docs/BASELINE_ENVIRONMENT.md)

```bash
bash majob.sh --pipeline
```

managed_cluster 的默认 pipeline 将学生权重保存在共享目录 `/workspace/baselineForLogKV/checkpoints/qwen8b-hattention/`。多节点启动时先检查其他节点能否读取节点 0 新写的文件。旧作业目录中仍可见的相应 CPT、蒸馏或转换产物按阶段优先级迁入共享目录并核对文件摘要，原文件保留。

验证保存与恢复时可运行 `bash majob.sh --pipeline --cpt-steps 10`。试跑复用已有蒸馏产物，权重保存到学生目录的 `cpt-10-steps/`，评测结果写入结果目录的同名子目录。再次提交相同命令会校验试跑产物并从评测继续。

重跑同一条 `bash majob.sh --pipeline` 即可。节点 0 优先校验 CPT 的文件哈希、权重结构、训练配置与完成步数，通过后所有节点直接进入评测。仅有完整蒸馏产物时，检查训练缓存后继续 CPT；已有产物校验失败时保留目录并报错。学生输出目录应在所有节点可见。


managed_cluster 提交一次作业，每个节点执行上面同一条命令。首节点转换权重、复用或准备 ProLong 缓存并做单卡 32K 试跑。随后所有节点依次参与蒸馏、32K CPT 和学生评测。阶段之间同步，任一节点返回错误，作业失败退出。多节点需要各节点可见的共享学生目录与数据目录，并开放额外的协调端口 `MASTER_PORT + 2`（默认 6002）。评测自动读取本次 `LLA_STUDENT_ROOT/cpt`。整条流水线从新的学生目录开始；已有成功产物时，可用下列单阶段入口从失败处继续。

```bash
bash majob.sh --train convert
# 已有匹配的 ProLong 32K 缓存可直接使用；仅有原始数据时运行下一行。
# bash majob.sh --train prepare
bash majob.sh --train probe
bash majob.sh --train distill
bash majob.sh --train cpt
bash majob.sh configs/qwen8b_hattention_eval.yaml
```

- **数据与路径。** 默认读取 [训练配置](configs/qwen8b_hattention_train.yaml) 指定的共享原始权重、数据和以 tokenizer 指纹命名的 32K LitData 缓存。可用 `LLA_QWEN_BASE`、`LLA_PROLONG_DATA`、`LLA_STUDENT_ROOT` 覆盖。缺少共享缓存时，`prepare` 将原始数据打包到学生输出目录的 `prolong_cache/`，训练自动读取。也可通过 `LLA_PROLONG_CACHE_ROOT` 指定包含 `litdata_<指纹>_ctx32768/` 的父目录。两种训练均从 32769-token 的块取数据，文档分词使用 `bos=False, eos=True`。CPT 将末尾 token 用作移位标签。
- **蒸馏。** 每个 32K 块切成八段连续 4K 输入。冻结原始 Qwen 教师及学生复用的主干。每层用同一份教师 pre-attention RMSNorm 输出，拟合教师经过输出投影、进入残差前的注意力输出。每层 MSE 除以教师输出的均方值，再对所有层取平均。初始配方为全局 batch 32、100 步、学习率 3e-4，共 13,107,200 输入 token。
- **CPT。** 从 `distill/` 加载学生，全参数做 next-token 交叉熵。默认全局 batch 32、1,200 步，共 1,258,291,200 输入 token。每卡 micro-batch 1，多卡 FSDP 分片；按实际 world size 累积梯度。默认 100 步 warmup 至 5e-6，余下步数余弦下降至 3e-6。训练使用 block 激活重算和分块词表投影。
- **试跑与成本。** `probe` 在一张卡上从转换权重取第一个真实 block，使用 32K 样本、冻结的 embedding 和 LM head 完成一次交叉熵更新，打印耗时与峰值 allocated/reserved 显存。这个数值只覆盖单层。训练预览用 `bash majob.sh --train cpt --print-config`，缩短蒸馏或 CPT 可加 `--steps 1`。使用独立输出根目录做试跑，每阶段成功后原子发布 checkpoint，重跑时会校验已有阶段的 SHA-256、权重结构与运行配置。校验通过即跳过该阶段，残缺或不匹配时停止并保留原文件。阶段内部中断需从上一阶段产物重新运行该阶段。
- **平台与结果。** 蒸馏和 CPT 每节点启动一次 `majob.sh`；转换、准备和单卡试跑仅在首节点执行。每个学生 checkpoint 包含模型、tokenizer、源权重与产物哈希、阶段记录。最终产物分别保留蒸馏和 CPT 的训练 token 与 GPU 小时。训练记录另含实际步数、token 数、各 rank 的输入流摘要、时间和峰值显存。评测默认读取 `checkpoints/qwen8b-hattention/cpt`，自定义根目录时设置 `LLA_CHECKPOINT_DIR="$LLA_STUDENT_ROOT/cpt"`。沿用分支的 32K 主线与六档长测；学生单独标记，dense KV 分母按 Qwen3-8B 的 36 层、8 个 KV 头、每头 128 维计算。所有长测使用同一个 32K CPT checkpoint。

本地使用微型 Qwen3 在 Python 3.12 CPU 环境复测了服务器对应的 PyTorch 2.11.0、Transformers 4.51.3、LitData 0.2.61、Hugging Face Hub 0.36.2 和 Safetensors 0.7.0。单进程与双进程 FSDP 均跑通蒸馏、CPT 和导出，评测适配器完成了一次打分和生成。宽度 16 的单层通过 32,768-token 前向、反向和参数更新。8B 权重、CUDA/NCCL 资源开销及 1M 任务得分仍需目标平台实测。

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

## 论文汇总表

`lla_bench.summary` 沿用参考版本的表头、指标映射和分组平均。可同时读两个模型的运行日志或带时间戳的结果 JSON，每行一个模型、共 458 列。六档 RULER、BABILong、LongBench v2、多轮问答、PPL 曲线及支持长度内的 InfiniteBench 都有独立列。仅传同一实验的文件，未测项目留空。和参考实现一样，各组平均按可用项计算，小样本或缺项结果需要在组间比较前核对覆盖范围。

```bash
# 各自将完整评测的 stdout/stderr 存为日志后
python -m lla_bench.summary --model HAttention hattention.log \
  --model LogKV logkv.log --output comparison.csv
```


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

`python -m pytest -q tests/test_native_hattention.py` 检查门控因果定义、任意 prefill 分块、跨层级进位、逐 token 解码、重置与字节计数。本分支开发时还用作者公开权重、原配 tokenizer、上游 `make_masked_H_matrix` 做了独立 FP32 前向核对；7 个位置共 224,000 个 logits，最大绝对误差 7.63e-6。CPU 小型状态流覆盖 1,049,600 个位置。使用作者权重在 CPU 上逐路跑通了 BoolQ 1 题（2 个选项）以及 32K `niah_single_1` 1 题（输入 32,585 token、生成 128 token、无截断），均产生 JSON、CSV、XLSX。两个样本均得 0 分，仅用于检查端到端执行。目标 GPU 上的全套质量、速度和显存峰值仍待实测。

```bash
python -m pytest -q tests/test_native_hattention.py tests/test_hattention_reporting.py
```

- 在 Python 3.10.20、PyTorch 2.11.0 CPU、lm-eval 0.4.11、datasets 4.8.4、transformers 4.51.3 上，五项回归测试通过。使用作者完整权重、原配 tokenizer，经 `LLA_EVAL_DEVICE=cpu` 和 `eval.sh` 完成 BoolQ 单样本的两个选项打分。
- 使用完整 FP32 权重，流式前向与作者 `make_masked_H_matrix` 的逐层矩阵实现比较，8 个位置共 256,000 个 logits，最大绝对误差为 1.14e-5。此核对在 CPU 上执行。
- 六档任务名、窗口、分阶段任务清单与固定参照提交逐档核对。使用小尺寸随机模型跑通 lm-eval 的真实 1M RULER 生成和 PG19 评分，前向分别消费 1,048,065 和 1,048,577 个 token，并导出长度分数、PPL、bits/byte、资源汇总和 XLSX。另用可精确求和的递推输入核对全部 1,049,600 个位置。这些测试验证长窗口和指标管线。
- A800 上的整卡峰值、速度以及公开权重的全量质量分需要在目标环境实测。请保留双方的检查点、数据、长度、样本数和实际执行设置。
