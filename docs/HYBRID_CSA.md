# Qwen3 Full + CSA4 baseline

本分支从原始 **Qwen3-1.7B-Base** 出发，把部分注意力层替换为学习式 CSA4，冻结原始参数，仅适配新增压缩器和 indexer。它是 DeepSeek CSA 的 **Qwen GQA 移植版本**，不是 DeepSeek V4 权重、整网架构或官方性能复现。训练后的有效模型为 `Base + adapter.pt`，不能称为与免训练对照权重相同。

## 1. 默认层排列

28 层使用 `[Full, CSA, CSA] × 9 + [Full]`：

- Full 层（从 0 开始）：`0,3,6,9,12,15,18,21,24,27`，共 10 层；
- 其余 18 层使用 CSA4；局部窗口 128，压缩条目 top-k 512；
- indexer 使用 4 个 query head、每 head 64 维、独立单流 compressed key；
- 该排列让完整注意力均匀出现，首尾保留 Full，最多连续两个压缩层，整网主 KV 约减半。

这是一项事先固定的成本与适配折中，**不是根据 NIAH 选出的最优布局**。不用捞针样本训练，不按捞针结果挑 checkpoint，不默认扫层比例。`full_layer_interval` 可用于明确提出的后续对照，默认不改。

## 2. CSA 机制及移植边界

算法依据为 [DeepSeek V4 技术报告 §2.3、§4.2](https://arxiv.org/html/2606.19348v1) 和 [Flash 官方推理源码](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/2b2bebc1776b2cf88ca36f0b5b6f816ae0f5892f/inference/model.py)（核对版本 `2b2bebc1776b2cf88ca36f0b5b6f816ae0f5892f`，2026-10-20）。

保留的关键机制：

1. stride 4；每个压缩条目覆盖前一块 4 token 和当前块 4 token。两组使用不同的 payload/gate 角色投影，第一块缺失的历史显式 mask。
2. 每通道可学习 gate 加四个相位的偏置，沿 8 个源 token 做 FP32 softmax；压缩率由 stride 决定，不由 top-k 决定。
3. 压缩 K 的 RoPE 锚点是当前块起点 `4j`，只有在块末 `4j+3` 已可见时才能被查询。未闭合块不生成压缩条目。
4. 当前 token 在内的 128 项原始局部 KV 与选中的压缩 KV 共用一次 softmax；二者内容重叠是有意保留的。
5. 先因果 mask，再选择 top-k，并保留有效位；历史不足时不能把 `-inf` 占位项当成有效缓存。
   ReLU 打分可能精确同分，本实现用 stable argsort 取前 k 项，同分固定选较早块；避免候选矩阵长度改变时 `torch.topk` 的不稳定平局影响分块结果。这是确定性实现规则，不是检索优化。
6. 所有已闭合压缩项和 index key 保持存储。top-k 512 只限制读取，不是缓存最多存 512 项。
7. 新 token 同样参与压缩；整段训练、任意分块 prefill、逐 token decode 遵守同一因果规则。

有意保留 Qwen 的部分：

| 部分 | 本实现 | 与官方 V4 的差异 |
|---|---|---|
| 主注意力 | Qwen 16 Q heads / 8 KV groups，独立 K/V | 没改成 V4 的共享 K=V MQA |
| payload | 原始 Qwen norm 后、RoPE 前 K/V，经两个可学习 per-head 方阵映射 | 以 identity 初始化以复用 Qwen 权重；不是官方完整 hidden→共享 KV 新投影 |
| gate | hidden→逐通道 gate，K/V 分开 | 保留学习式 overlap pooling，适配 GQA 双流 |
| RoPE | 原始 Qwen full RoPE 与 theta；压缩 K 在块起点旋转 | 不移植 V4 部分 RoPE、YaRN、专用压缩 theta |
| indexer | 独立小型 query/key/gate，ReLU 打分，注意力分布 KL 监督 | 不移植 V4 低秩 query 通路、FP8、量化或 Hadamard kernel |
| 层混合 | Full + CSA4 | 没有 HCA128、MoE、mHC、attention sink |

因此该实验回答的是“Qwen 在这个 CSA4/Full 配置和适配预算下的捞针表现”，不能将结果归因于 DeepSeek 官方模型。

## 3. 理论 KV 口径

按用户要求，只统计理论条目/元素，不测缓存分配字节或实际显存。设实际已前向 token 数为 `T`，Full 层数 `F=10`，CSA 层数 `C=18`，总层数 `L=28`，窗口 `W=128`。

主 KV 保留率：

```text
R_KV(T) = [F*T + C*(floor(T/4) + min(T,W))] / (L*T)
压缩倍数 = 1 / R_KV
节省比例 = 1 - R_KV
```

同一 KV group 的一对 K/V 算一个条目；分母为同一 Qwen 几何、同一 T 的全 Full 缓存。因为压缩项与原始项宽度相同，条目比就是主 KV 元素比。不能再额外除以 GQA 的 8 个 group，也不能把 stride4 称为整网 4 倍压缩。

| 32768 token，batch=1 | 理论值 |
|---|---:|
| Full / CSA 层数 | 10 / 18 |
| 每 CSA 层闭合压缩项 / 局部项 | 8192 / 128 |
| 主 KV 保留率 | **52.036830%** |
| 主 KV 节省比例 | **47.963170%** |
| 主 KV 压缩倍数 | **1.921716×** |
| 无限长主 KV 保留率（忽略固定窗口） | 51.785714% |
| 主 KV + index key 元素保留率 | 52.539063% |
| 再含压缩器尾部状态的元素保留率 | 52.554757% |

辅助元素单列：index key 为 `C*floor(T/4)*64` 个标量；压缩器保留上一完整块与当前余块，`tail = T-max(0,(floor(T/4)-1)*4)`，最多 7 token，每 token 保存 hidden 和 pre-RoPE K/V，共 `2048+2*8*128` 个标量。后者是本实现的流式工作状态，不属于主 attention KV；没有把它悄悄藏进主 KV 比例。这里按元素计，忽略 dtype、分配余量、临时矩阵、权重和共享 RoPE。

评测记录每个请求**实际前向**长度，包括已送入模型的生成 token，不含尚未前向的最终输出 token。短于 128 的请求可能因局部项与压缩项重叠而超过 100% 保留率，这是正确计数。汇总使用 `sum(retained)/sum(dense)`，不平均请求百分比。32K 表是指定长度的理论值，不冒充所有实际请求的平均。

```bash
python -m hybrid_csa.budget --tokens 32768
```

## 4. 适配训练

独立训练配置：`configs/hybrid_csa/train_qwen1.7b_32k.yaml`。默认训练长度 4096、全局 batch 8；总 160 个 optimizer step，**5,242,880 token exposures**。这是可运行的初始预算，未声称这些步数足以恢复能力。

| 阶段 | 步数 | 训练参数 | 主注意力 / 目标 |
|---|---:|---|---|
| compressor | 64 | 压缩器 | local + 全部闭合压缩项；通用 LM CE |
| indexer | 32 | indexer | 同上；对 detached 主注意力的 compressed 分布做 KL |
| sparse | 64 | 压缩器和 indexer | local + top-k；LM CE + 选中项上的 index KL |

原始 Qwen 参数始终冻结；离散 top-k 不会把 LM 梯度传给 indexer，不能省略 indexer 监督。训练长度大于 `512*4=2048`，才真正进入 top-k 截断。新增参数和 gate/attention logits/softmax 保持 FP32；主干 BF16，按块计算 CE，使用 activation checkpointing。

真实 Qwen 几何的静态参数计数：压缩器 **160,727,040**，indexer **19,031,040**，共 **179,758,080** 个新增可训练参数。这里“小量适配”指较少训练 token/步数，不是 LoRA 级别的小参数量。运行 manifest 会记录实际配置下的参数数目。

语料仅使用指定本地 JSON/JSONL/Parquet；语料目录中 LitData 写出的目录（`index.json` 含 `chunks` 与 `config`，如 SemanticLogKV 放在共享 prolong 目录里的 `litdata_<key>_ctx32768/`）是别的作业的打包结果，不当作文档，整个跳过并记入数据 manifest 的 `skipped_litdata_outputs`。字段优先级为 text/chapter/content/code，messages 则显式拼成角色段落；空或不兼容记录直接报错。BOS 关闭、每文档追加 EOS，跨文档打包。记录原始文件哈希、tokenizer 哈希、prepare 参数和 index 哈希。每阶段重新开始数据迭代和 AdamW；token 数是包含阶段重读的 exposure 数，**不是独立语料 token 数**。阶段内数据不足会失败，不静默循环。

训练与评测都在已有的 torch218 中运行，不安装本项目或任何包。训练只用 PyTorch、LitData 与 PyArrow，torch218 已有
（2026-10-16 快照：torch 2.11.0+cu128、litdata 0.2.61、pyarrow 23.0.1、lightning 2.6.1）；精确版本写入训练 manifest。
`scripts/hybrid_csa_train.sh` 与评测脚本使用同一个解释器（`HYBRID_CSA_PYTHON` → `BASELINE_EVAL_PYTHON` → `BASELINE_PYTHON` →
torch218 绝对路径）；设置了旧的 `HYBRID_CSA_TRAIN_PYTHON`/`BASELINE_TRAIN_PYTHON` 时报错，不会被悄悄忽略。

```bash
python -m hybrid_csa.train --print-config
# 一次性数据准备，必须单进程
bash scripts/hybrid_csa_train.sh configs/hybrid_csa/train_qwen1.7b_32k.yaml --prepare-data
# GPU 作业，每节点执行一次
bash scripts/hybrid_csa_train.sh
```

路径环境变量：`HYBRID_CSA_BASE_CHECKPOINT`、`HYBRID_CSA_DATA_DIR`、`HYBRID_CSA_TOKENIZED_DIR`、`HYBRID_CSA_TRAIN_OUTPUT`。具体默认值以 `--print-config` 为准。输出目录必须是新目录，分阶段保存 `compressor.pt`、`indexer.pt`、`sparse.pt`、日志和 manifest；记录实际步数、token、墙钟、GPU 小时、代码与环境身份。GPU 小时只表示运行耗时×卡数，不是计费成本。

`--adapter` 配合 `--start-stage indexer|sparse` 可从已训练阶段继续权重；不恢复 optimizer 或精确数据位置，manifest 明确记录。默认评测必须提供经过 sparse 阶段训练的 adapter，并严格核对 Base、结构和 tokenizer 指纹；初始化均值池化不会被当成正式 CSA 结果。

## 5. 固定 32K 评测

项目内私有 vendor 固定 SemanticLogKV 来源 commit `4fd97c78e04f0ec96ad6aed18df9e9cf440c6090`；不读取相邻项目源码或 YAML。沿用其 loglikelihood、rolling、generate、分词和评分接口。当前共享生成协议只按 EOS/最大长度停止，不执行 until 字符串；本分支不单侧改变这项语义。

在已有 torch218 环境、项目根执行：

```bash
python -m hybrid_csa.evaluate --suite niah --print-config
# 首次先少量真实样本，确认产物与环境可用；不会隐式训练
bash majob.sh --adapter outputs/hybrid_csa/adaptation/sparse.pt --limit 4 --output outputs/csa-smoke
# 确认后才移除 --limit，main 47 入口和 NIAH 3 入口分阶段
bash majob.sh --adapter outputs/hybrid_csa/adaptation/sparse.pt --output outputs/csa-full
```

只跑 NIAH：`bash scripts/hybrid_csa_evaluate.sh --suite niah --adapter /path/to/sparse.pt`。NIAH 长度只允许 `[32768]`，完整窗口上限 32768，预留生成预算。

同入口 `--backend dense` 跑原始 Base，`--backend logkv --model /path/to/logkv/checkpoint` 跑对照，不需要 CSA adapter。任务、tokenizer 和逐样本输入必须一致。使用 `python -m hybrid_csa.compare_results LEFT.json RIGHT.json` 核对协议与输入；方法代码和 adapter 身份单列，不要求两种算法实现相同。旧结果缺失可核对的协议指纹时拒绝自动判定一致，不复用旧分数。

平台入口 `majob.sh` 默认属于本分支的 HybridCSA；旧 ChunkKV 平台入口保存为 `scripts/chunkkv_majob.sh`。

### 5.1 平台一体化入口

**默认完整流程：检查训练和评测依赖 → 准备/核验通用语料 → compressor/indexer/sparse 三阶段训练 → 校验 sparse adapter → main → NIAH。**
所有阶段直接调用同一个 torch218 解释器，不在作业中激活 conda 或安装/升级包。`--mode train` 只准备与训练；`--mode eval` 只评测。
为兼容既有用法，未写 mode 但提供 `--adapter`，或使用 `--backend dense|logkv`，自动选择 eval；评测 CLI 自身永远不训练。

```bash
# 训练与评测共用 torch218；缺省即此路径，可显式固定
export HYBRID_CSA_PYTHON=/home/ma-user/anaconda3/envs/torch218/bin/python
# 默认路径已对齐现有资产；需要时覆盖，环境路径必须为绝对路径
export HYBRID_CSA_BASE_CHECKPOINT=/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/
export HYBRID_CSA_DATA_DIR=/home/ma-user/work/bucket-pangu-green/lihourun/longcontext/prolong/data

# 不导入 torch/lm_eval，不访问权重，不建输出目录
bash majob.sh --print-config
# 每节点执行同一命令一次，平台节点变量自动解析；输出见下表"本次输出"
bash majob.sh
# 已训练产物的少样本检查；去掉 --limit 才是完整评测
bash majob.sh --mode eval --adapter /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4/train/sparse.pt \
  --limit 4 --output /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4-eval-smoke
```

**不要向 torch218 安装任何 extra，包括 `[eval]`**；训练不需要额外依赖。
`--limit` 只限制评测样本，完整流程中的训练仍执行配置的 160 步。

| 项目 | 优先级 / 约定 |
|---|---|
| Python（所有阶段） | `--eval-python` → `HYBRID_CSA_PYTHON` → `BASELINE_EVAL_PYTHON` → `BASELINE_PYTHON` → 已有 torch218 绝对路径；与此前评测解释器的规则相同。`--train-python`、`HYBRID_CSA_TRAIN_PYTHON`、`BASELINE_TRAIN_PYTHON` 已取消，出现即报错 |
| Base / 语料 | `--base-checkpoint-dir` / `--data-dir` → `HYBRID_CSA_*` → `BASELINE_BASE_CHECKPOINT` / `BASELINE_DATA_DIR` → 项目 YAML |
| 分词缓存 | `--tokenized-dir` → `HYBRID_CSA_TOKENIZED_DIR` → 训练 YAML（当前 `/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4/tokenized`）。放在持久共享存储上：第一次运行准备一次（全量 241 个分片），之后的运行校验后直接复用。可以位于输出目录中、与结果并列，但不能在每次运行会删除的 `train/`、`eval/`、`main/`、`niah/` 里；与原始语料和评测 HF 缓存分开 |
| 本次输出 | `--output` → `HYBRID_CSA_OUTPUT_ROOT` → `BASELINE_OUTPUT_ROOT` → `configs/hybrid_csa/majob.yaml` 的 `output`（当前 `/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4`）；每次运行覆盖，见下文 |
| 配置 | 可选位置参数选择项目内实验 YAML；`--train-config`、`--hybrid-csa-config` 选择项目内训练/方法 YAML；结构设置必须一致 |
| 相对路径 | shell 入口先切到项目根；直接 Python CLI 的路径参数相对于调用者 cwd，YAML 内的资产相对路径相对于项目根 |
| HF 路径 | `HYBRID_CSA_HF_CACHE` → `BASELINE_HF_CACHE` → `HF_HOME` → 默认根；显式 HF 子目录、RULER、NLTK 路径保留 |
| GPUs / 节点 / rank | `MA_NUM_GPUS` → `NPROC_PER_NODE` → `LOCAL_WORLD_SIZE` → 实测；`MA_NUM_HOSTS` → `NNODES` → `GROUP_WORLD_SIZE`；`VC_TASK_INDEX` → `NODE_RANK` → `GROUP_RANK` |
| 地址 / 端口 | 显式 `MASTER_ADDR` 优先，否则平台 DNS；多节点禁止 localhost。训练基准端口 6000，main +1、NIAH +2、prepare +3；支持各阶段端口变量；单节点使用空闲端口 |

配置预览中 GPU 数为空表示运行时检测。同一解释器分别做训练与评测两项预检（各自导入所需模块，实际执行 `torch.cuda.init()` 与 BF16 检查），取可见卡数，并在训练前核验全局 batch 可整除总 GPU 数。
默认全局 batch 8、每卡 micro batch 1，训练进程总数须整除 8。卡数多于配方可用时，训练自动只用每节点能整除的最多卡数（如 1 节点 16 卡或 2 节点各 8 卡时共用 8 卡，6 卡时用 4 卡），梯度累积随之调整，全局 batch、步数与 token 预算不变；评测仍用全部卡。启动日志与 `pipeline_node_<rank>.json` 的 `topology.train_gpus` 记录实际训练卡数。节点数不能整除 8（如 3 节点）时报错；要改用更多卡训练，须显式修改训练 YAML 的全局 batch，并重新记录训练 token 成本。

准备阶段用 Gloo 每节点启动一个进程，只有全局 rank 0 调用分词准备；各节点核对配置一致，接收成功/失败并检查共享文件可读。已存在完整缓存时核验 tokenizer、context、index 哈希与原始语料的文件清单和大小后复用（各文件 SHA-256 只在首次准备时计算并记入 manifest，复用时不再重读 100+ GB 语料；同名同大小的内容改动因此不会被发现，共享语料按只读资产对待）；没有完成标记（数据 manifest）的目录是此前中断的准备，自动删除后重做；已完成的缓存不自动删除。
多节点要求代码/资产位于一致路径，分词缓存和输出目录共享可见。训练完成并退出后，评测解释器使用 `weights_only=True` 校验产物的 Base/tokenizer 指纹、方法设置和 sparse 训练记录；真正载入模型时再严格核验参数键和形状。

本次输出为：

```text
<output>/pipeline_node_<rank>.json   # 解释器（plan.python）、两项预检的精确包版本、解析配置、阶段命令、起止时间及失败信息
<output>/train/                     # resolved_train.yaml、manifest、metrics.jsonl、三个 adapter
<output>/eval/main/                 # resolved_eval、结果和逐样本输入
<output>/eval/niah/
```

eval 模式结果直接位于 `<output>/main` 和 `<output>/niah`。**同一输出目录每次运行都会被覆盖**：环境预检通过后、开始准备与训练前，
节点 0 删除上次运行的 `train/`、`eval/`、`main/`、`niah/`（只评测时保留 `train/`，要评测的 adapter 若在被删目录里则报错停止），
每个节点替换自己的 `pipeline_node_<rank>.json`，并删掉本次不存在的节点留下的记录；目录中的其他文件和分词缓存不动。
因此目录里只会有本次运行的结果；需要保留的旧结果须在下次运行前自行复制。不会从历史文件中选 adapter。

### 5.2 当前评测环境与比较边界

按当前环境清单适配 `lm-eval==0.4.11`、`datasets==4.8.4`，启动最低要求分别为 `0.4.11`、`4.8.4`，PyArrow 最低 21；范围检查不代表不同版本分数可比较。
0.4.9 的 LongBench 数据源、prompt 和采样配置不同，不再作为本分支的评测协议。每次结果记录精确版本，比较工具拒绝版本不一致。
不引入 accelerate、sentencepiece 或外部 editable litgpt 依赖。评测保持离线，map/filter 使用进程私有临时缓存，默认 `TMPDIR=/tmp`。

本次 vendor 仅额外修改 HF 环境赋值为保留显式路径；prompt、任务请求、生成和评分函数未改动，来源与逐文件哈希更新在 `SOURCE.json`。
完整源码协议指纹因此变化，**没有把该差异从比较指纹中抹掉**：对照应使用本分支同一入口的 dense/logkv 后端重跑，不能直接沿用旧分支结果。
环境预检只验证包导入、CUDA 和基础文件，不证明 50 个入口的数据齐全；真实离线读取失败会明确报错并停止。
预检记录各包精确版本；必需包缺失即报错。xlsxwriter 是可选的：参考评测器只用它写 `inference_metrics.xlsx`，缺失时打印警告并跳过该文件，
评测结果与逐样本记录不受影响；预检把它记为 `null` 并在启动日志中提示（2026-10-21 平台 torch218 上即如此）。

## 6. 易错点与验证范围

- **不能先 full prefill 再压缩。** CSA 从 prefill 第一段就运行压缩注意力，否则捞针的信息已被 full attention 传到末尾，测到的不是本机制。
- **不能对已旋转 K 再池化/旋转。** 本实现保留 pre-RoPE 源，压缩后在块起点旋转；可见性另用块末判断。
- **不能丢弃跨分块的源状态。** 保存最多 7 token 的 overlap 源和局部窗口，覆盖非 4 整除的 chunk。
- **不能只写 `.float()` 就假定 FP32。** autocast 会再次降低 einsum 精度；gate 和 CSA attention 显式禁用 autocast。
- **不能给训练预算充分性背书。** 4K 适配后 32K 检索能力须由固定评测得出；结果低可能包含迁移/预算不足的影响。
- **性能限制。** 当前为纯 PyTorch 正确性基线，按 query chunk 限制临时工作量；没有 V4 专用融合内核，32K 速度未经验证。不要把时间开销当成 CSA 算法的最优性能。

2026-10-20 训练移入 torch218 后的验证（开发容器 CPU 上的 torch218 副本：Python 3.10.20、torch 2.11.0、lightning 2.6.1、
litdata 0.2.61、pyarrow 23.0.1、lm-eval 0.4.11、datasets 4.8.4、transformers 4.51.3、tokenizers 0.21.4，与环境清单第 1 节一致；没有 GPU）：

- 改动前（`0d7ee4c`）的全部测试在该副本中已通过（141 passed），包括真实小模型三阶段反传、LitData 打包读取与 checkpoint 测试，
  说明训练代码在 torch218 的包版本下可运行；改动只涉及解释器选择、依赖声明与文档，训练、评测和模型代码未改。
- 改动后同一副本：143 passed（新增旧训练解释器设置被拒绝的 2 项）。
- 用同一个假解释器分别运行改动前后的 `majob.sh`：完整流程、`--mode eval --adapter`、`--backend dense` 三种模式下，每个阶段的命令、
  参数、工作目录与 HF/CUDA/拓扑相关环境变量逐字相同；评测阶段的接口、输出位置和环境没有变化。
- 未验证：真实 CUDA 上的训练（`hybrid_csa.train` 要求 CUDA/BF16，测试只覆盖 CPU 小模型）、多 GPU/多节点通信、原始 1.7B 与 32K 评测。

2026-10-20 本机验证（训练移入 torch218 之前）：

```bash
python -m pytest -q tests/hybrid_csa tests/chunkkv/test_evaluate.py --disable-warnings
# 95 passed, 1 warning
```

- macOS CPU 隔离环境：Python 3.12.7、torch 2.7.0、lightning 2.6.5、litdata 0.2.59、lm-eval 0.4.11、datasets 4.8.4、PyArrow 21.0.0、transformers 4.52.3、tokenizers 0.21.1、pytest 9.1.1。没有改动服务器 torch218；这是局部接口测试环境，并非服务器完整 freeze 副本，不能据此宣称目标环境全量兼容。
- HybridCSA 测试覆盖：全 Full 与原 GPT 一致、因果未来扰动、三阶段整段/不规则分块/逐 token 对齐、跨局部窗口、同分 indexer、checkpoint 梯度一致、CPU BF16 主干反传、冻结边界、真实本地文本分词与 LitData 打包读取、adapter 身份校验、预算公式、独立配置/导入。
- 一体化入口以真实 shell/Python 启动器和模拟阶段解释器验证：跨目录调用、两个解释器的选择、prepare 单进程/训练多进程、节点 rank 和各阶段端口、环境/准备/训练/校验/main 失败停止、缺少 adapter 拒绝评测、历史输出拒绝复用。此项不等于真实多节点通信已验证。
- 30 个原 ChunkKV 评测/启动器回归通过；其平台入口移至 `scripts/chunkkv_majob.sh`。
- 真实 `lm-eval 0.4.11` 包下，小模型训练并保存 adapter 后通过 `generate_until`、`loglikelihood`、`loglikelihood_rolling` 三类请求；没有跳过项。CPU DataLoader 的 pin-memory 提示为一条 warning。
- `bash -n`、`git diff --check` 通过；私有 vendor 哈希核对通过。命名空间标准化后，评测全部函数和类的 AST 与固定本地参照一致；模块级 HF 路径赋值差异已记录。真实 1.7B 几何只用 meta device 核对结构和参数数目，没有加载权重。

目标 CUDA/BF16、原始 1.7B 权重、离线真实评测样本、完整 32K 和多 GPU 作业仍需在服务器验证。本分支未运行真实适配训练，未产出捞针分数。
