# Qwen3-8B-Base 初始化的 Mamba-2 baseline

每层用 Mamba-2 SSD 读取历史，保留 Qwen3 的词表、嵌入、SwiGLU MLP、块内 RMSNorm、末层 RMSNorm 和输出头。36 层全部替换注意力。评测代码、任务配置和启动器随分支提供；可直接复用 LogKV 的模型目录、ProLong chunks 和离线评测数据。[上游代码版本与许可证](mamba2/SOURCE.md)

| 参数（含词嵌入与独立输出头） | 数量 |
|---|---:|
| Qwen3-8B-Base | 8,190,735,360 |
| 本分支 Mamba-2 | 8,199,164,400 |
| 原样复制自 Qwen | 6,680,776,704 |
| 新初始化的 Mamba-2 参数 | 1,518,387,696 |

模型宽 4096，MLP 中间维 12288。SSD 每层内部宽 3328，有 52 个 64 维 head、2 组输入依赖的 B/C、64 维状态、宽度 4 的因果卷积和门控分组 RMSNorm。Q/K/V、注意力输出投影、Q/K norm 和 RoPE 的语义与 SSD 不同，转换器只迁移共享张量。运行 `cd mamba2 && python prepare.py --count-only` 可重新清点。初始化后可以跑消融；正式对比应评训练后的 checkpoint 并注明适配成本。

对长度 N，完整 prefill 计算量 O(N)，单步 decode 计算量及推理持久状态均为 O(1)。batch=1、bf16 卷积历史和 fp32 SSD 状态时，36 层合计 31,444,992 字节。状态不包括权重、输入、输出和临时激活。SSD 矩阵仅在固定大小的块内构建。使用现有 torch 算子，无需编译或安装 mamba-ssm。公开融合内核的性能需单独测量。

## 准备和训练

下列命令从分支根目录执行。managed_cluster 入口沿用 torch218、CUDA 12.8、NCCL 和原离线缓存环境。已备好的 Qwen3-8B-Base 可以是 LitGPT 的 `lit_model.pth` 或 HF safetensors。转换在新目录创建文件并记录随机种子和复制参数数量。

```bash
# 首次初始化（在准备节点执行一次）
bash majob.sh --prepare
# 在 GPU 作业上训练，成功后接完整评测
bash majob.sh --train && bash eval.sh
```

若缺底模，在联网准备节点运行下列命令。下载器固定本次解析到的 revision，按普通文件写入，核对大小及可用的 LFS SHA256。训练和评测本身离线运行。已有本地权重时可跳过下载。

```bash
(cd mamba2 && bash diag.sh fetch_qwen.py --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --out "${QWEN_BASE:-/workspace/checkpoints/Qwen/Qwen3-8B-Base}")
```

CPT 沿用 8B LogKV 的 ProLong 数据路径、32768 上下文、全局 batch 128、micro batch 1、2000 步和初始学习率 5e-6。完整预算为 8,388,608,000 个输入 token，实际完成步数、吞吐和耗时以 checkpoint 元数据和 `training_metrics.jsonl` 为准。训练全模型，使用 Fabric FSDP、activation checkpointing 和分块交叉熵。这是与原实验配平的起始配方，其收敛和长上下文质量有待实测。[训练配置](mamba2/exp/cpt.yaml)

若 ProLong 已按同一 tokenizer 和上下文切好，可直接复用。路径不同时设置 `PROLONG_DATA`。只有原始语料时，先在可写的数据目录准备 chunks。

```bash
export PROLONG_DATA=/workspace/longcontext/prolong/data
(cd mamba2 && bash diag.sh data.py --tokenizer_dir "${QWEN_BASE:-/workspace/checkpoints/Qwen/Qwen3-8B-Base}" --data_dir "$PROLONG_DATA" --context_length 32768)
```

## 评测和对照

```bash
# 主线后跑六档长度，默认读取 MAMBA2_CKPT，长文本每任务上限 100 条
set -o pipefail
bash eval.sh | tee mamba2-full.log

# 指定任务做小检查
LIMIT=2 EVAL_LENGTHS=none bash eval.sh exp/eval_full.yaml piqa niah_single_1

# 单独复评 512K 和 1M
EVAL_LENGTHS=512k,1m bash eval.sh

# 预览独立长度阶段的任务、窗口和输出目录
bash majob.sh --plan
```

每个节点执行一次启动器。普通生成、选择题、多轮共享状态、rolling PPL、跨 rank 汇集使用同一版 lm-eval 协议。主线为默认 49 个 main 入口和 20 个 NIAH 入口，后者取 1K–32K 六种长度。之后按下表评测，窗口额外保留 1024 tokens，并剔除主线已测的任务和长度组合。沿用 LogKV 的扩展 RULER 样本协议；Mamba-2 使用相同窗口，其结构不使用 RoPE/YaRN。

| 任务和指标 | 长度 |
|---|---|
| RULER（含 NIAH），按任务及长度报告召回或任务分数 | 32K、64K、128K、256K、512K、1M；SQuAD 至 128K |
| BABILong qa1–qa5、LongBench v2，各自的准确率及分组 | 六档 |
| NIAH 多轮，同一状态连续回答，报告总准确率及 turn1–turn4 | 六档 |
| 长文末段 PPL、bits/byte，长书的多位置 PPL 曲线 | 六档 |
| InfiniteBench 的 10 项任务，各自官方评分 | 32K 至 256K |

默认完整计划与上游一样，在数据缺失时会报告并跳过可选的 PPL、books 或 InfiniteBench。生成最终对比表时，两侧指向相同的 `LONG_PPL_DATA`、`LONG_PPL_BOOKS`、`LONG_PPL_DOCS`、`INFINITEBENCH_DIR` 和 tokenizer。可在完整六档显式设置 `EVAL_SUITES=ruler,babilong,ppl,books,turns,longbench2,infinitebench`，并先运行下方数据检查。单独选超长档按上游默认运行 RULER、BABILong、books、LongBench v2，其他长文本套件可显式选择。LongBench v2 缺数据会报错。[六档评测配置](mamba2/exp/eval_full.yaml) · [上游 512K/1M 协议](mamba2/docs/eval-512k-1m.md)

在联网节点补齐长文本数据时，使用分支内的原下载工具。下例使用已有的评测长文和书籍，可用 `--ppl-hub` / `--ppl-books-hub` 指定与对照一致的来源。默认采用平台的无证书校验方式，环境变量由 `diag.sh` 设置。主线常识及 LongBench v1 等任务继续复用 torch218 的原离线缓存。

```bash
(cd mamba2 && bash diag.sh unused/fetch_long_eval_data.py \
  --lengths 32k,64k,128k,256k,512k,1m --longbench2 --infinitebench --ruler-essays --ruler-cwe \
  --ppl-data "$LONG_PPL_DATA" --ppl-books-data "$LONG_PPL_BOOKS" --tokenizer "${QWEN_BASE:-/workspace/checkpoints/Qwen/Qwen3-8B-Base}")
(cd mamba2 && bash diag.sh unused/check_long_eval_data.py \
  --lengths 32k,64k,128k,256k,512k,1m --tokenizer "${QWEN_BASE:-/workspace/checkpoints/Qwen/Qwen3-8B-Base}")
```

结果写入 `MAMBA2_RESULTS` 或配置中的独立目录，按长度分子目录。JSON 记录样本、精确依赖版本、模型配置、参数量、TTFT、prefill/decode/forward 吞吐和 CUDA allocated/reserved 峰值。SSD 和卷积状态按实际分配计入 `cache_budget.persistent_cache_bytes`。KV 槽位压缩率记为 `null`；内存对照可使用双方的持久字节和 CUDA 峰值。固定 `LOGKV_EVAL_PREFILL_CHUNK` 与 `LOGKV_EVAL_MLP_CHUNK` 并记录实际设置，便于解释速度差异；Mamba-2 的外层 prefill 分块默认 2048。[参考指标口径](mamba2/docs/evaluation-metrics.md)

```bash
# 对同一阶段的两份完整结果核对样本、prompt、target 和生成参数
python mamba2/compare_inputs.py "$LOGKV_STAGE_JSON" "$MAMBA2_STAGE_JSON"

# 从两侧实际日志汇总主线和六档长文本表
python mamba2/unused/parse_lmeval_table.py \
  --log LogKV logkv-full.log --log Mamba2 mamba2-full.log > comparison.csv
```

汇总表沿用上游的 `AVG_all` 和分档列。部分运行会留空，分组均值只平均已有值，比较时需核对两侧的任务与样本覆盖。分档 RULER、BABILong、PPL、逐轮与 InfiniteBench 可单独比较。

## 验证范围

在 CPU PyTorch 2.11.0、Transformers 4.51.3、lm-eval 0.4.11、datasets 4.8.4 下验证了 SSD 的前向和梯度、跨块状态与重置、单步优化、两种权重格式和四类评测请求；单组 mixer 与 Transformers 4.51.3 的 Mamba2Mixer 比对。启动器的 14 个阶段经 dry run 与固定版本的 LogKV 8B 完整评测逐项核对了任务、顺序、窗口和样本上限。

端到端使用真实 Qwen tokenizer、小型随机模型和正式 `niah_single_1`，在 32K、1M 各跑一条。实际 prompt 长度分别为 32,478 和 1,047,945，均无截断，每条生成 128 tokens，完成 JSON、CSV 和 XLSX 导出。本次没有目标 CUDA 机器、8B CPT 产物和完整共享语料，8B 模型的训练质量、GPU 性能及全套任务分数待测。复现本地回归可运行 `python -m pytest -q`。
