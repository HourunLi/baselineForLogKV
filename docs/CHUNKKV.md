# ChunkKV baseline：实现与 SemanticLogKV 对齐口径

更新：2026-10-15。分支：`chunkKV`（自 `DMS` 创建，DMS 代码原样保留）。论文：
[ChunkKV, arXiv:2502.00299v5](https://arxiv.org/html/2502.00299v5)。

ChunkKV 免训练。本项目独立维护实验 YAML 和必要的模型、训练、评测代码，直接读取与
SemanticLogKV CPT 起点相同的 Base `lit_model.pth`，不做 HF 转换。运行代码快照位于
`chunkkv/_vendor/semanticlogkv/`，采用项目内限定导入，不依赖相邻 `litgpt` 安装或另一项目的源码与 YAML。
快照来源、改动与许可证见该目录的 `README.md`、`SOURCE.json`、`LICENSE.md`。
运行环境要求 Python 3.10–3.13、PyTorch >= 2.7；模型推理和训练仍需相应运行依赖与实际权重、数据。

## 项目与配置边界

| 路径 | 用途 |
|---|---|
| `/data/lihourun/baselineForLogKV/chunkKV` | ChunkKV 项目根，以下命令均从此目录执行 |
| `/data/lihourun/semanticLogKV` | 独立的 SemanticLogKV 对照项目，ChunkKV 运行不要求其存在 |
| `configs/chunkkv/experiment_qwen1.7b_32k.yaml` | 自有平铺实验配置，保存 Base、数据、32K 上下文与训练/LogKV 参数，不继承外部 `exp/` YAML |
| `configs/chunkkv/qwen3_1.7b_32k.yaml` | ChunkKV 的选块、预算和层间复用参数 |
| `configs/chunkkv/eval_semantic_default.yaml` | Semantic 当前默认评测脚本的本地任务快照：47 个主任务入口和 3 个 NIAH，后者仅评测 32768 tokens |
| `configs/chunkkv/dense_cpt_qwen1.7b_32k.yaml` | 可选 dense CPT 配置，`experiment_config` 引用同目录的自有实验配置 |
| `outputs/` | 本项目评测与 dense CPT 的默认输出 |

评测用 `--experiment-config` 指定实验配置；旧 `--semantic-config` 只保留为这一参数的别名，
不再定位外部项目。`--litgpt-root` 已移除。实验中的相对资产路径以 ChunkKV 项目根为基准；
dense 配置中的 `experiment_config` 相对其配置文件目录解析。

项目地址与数据地址分别管理：没有证据表明原始权重、训练数据或缓存已迁移，所以仍保留现有
`/home/ma-user/...` 默认位置。迁移这些资产后，分别设置 `CHUNKKV_BASE_CHECKPOINT`、
`CHUNKKV_DATA_DIR`、`CHUNKKV_HF_CACHE`，或修改自有配置。比较 SemanticLogKV 的训练权重时，
用 `--model` 指定实际 checkpoint，不能直接把其项目根当作权重目录。

## 算法与论文的对应

| 论文 | 实现（`chunkkv/core.py`、`chunkkv/model.py`） |
|---|---|
| Alg.1 观察窗打分 | prompt 末尾 `window_size` 个 query 对全部 prompt key 做因果 softmax（模型自身 scale），对窗内 query 与**全部 query head** 求和，每层得到一组 token 分数 |
| Alg.1 块分数 `A_i = Σ_j A_{:,j}` | 窗口之前的前缀从位置 0 起按 `chunk_size` 切块，块分数为块内 token 分数之和；最后不足一块的部分按实际 token 求和 |
| Alg.1 Top-k 与拼接 | 选 `(budget − window) // chunk_size` 个最高分块（稳定排序，分数相同取靠前的块），按原顺序保留原始 K/V，再接上完整观察窗 |
| Alg.2 层间索引复用 | `reuse_layers = N`：仅第 `0, N, 2N, …` 层打分，其后 `N−1` 层复用其索引；`1` 表示不复用 |
| 压缩时机 | prompt 先完整 dense prefill（prompt 自身输出不受影响），每层算完 attention 后压缩一次；之后生成的 token 不再压缩，直接追加 |

同一层所有 KV head 共用一组块（跨 head 求和），与 KVPress `ChunkKVPress` 一致，也是层间索引复用的
前提。保留的是原始 K/V，不是块的均值摘要。

### 需要说明的取值与差异

| 项 | 取值 | 依据 |
|---|---|---|
| `chunk_size` | 10 | 论文默认块长 |
| `window_size` | 64 | 本环境无法访问 arXiv 原文核对观察窗长度；取论文所指 KVPress 实现中 SnapKV 打分器的默认值。可用 `--window-size` 改 |
| 块分数 | 求和、无 pooling | 按 Alg.1；KVPress 的 SnapKV 打分器另有 kernel 5 的 avg-pool，块内求均值，本实现未采用 |
| 观察窗 | 不参与块竞争，单独完整保留一次 | 等价于 Alg.1 "用观察窗替换所选结果的最后 w 个位置"，但不产生重复条目；预算中不足一块的余量（< `chunk_size`）不使用 |
| `budget` | 每层每个 KV head 的 prompt 条目数，含观察窗（论文的 `L_max`） | prompt 不超过预算时不压缩 |
| `reuse_layers` | 默认 1 | 论文主精度结果不依赖复用；复用是加速手段，按需设 2 或 3 |

论文中的其他实现细节（如原始实验代码的 kernel/批处理优化）未复现；计时只反映本实现。

## 与 SemanticLogKV 对齐的评测接口

`scripts/chunkkv_evaluate.sh` → `chunkkv.evaluate` → 私有模块实例中的本地快照 `eval.main`。
`ChunkKVLM` 继承快照中的 `LogKVLM`：构造时调用其构造函数（dense 模式）加载 tokenizer、
`model_config.yaml` 与权重，再把每层 `CausalSelfAttention` 包装为 ChunkKV 缓存；
`_score_tokens / loglikelihood / loglikelihood_rolling / generate_until` 全部直接继承。
同一入口也提供 `--backend logkv`（SemanticLogKV 本身）和 `--backend dense`（同模型 Full KV），
三者的结果 JSON 都带 `comparison_contract`，用 `chunkkv.compare_results` 核对。

| 项目 | 对齐方式 |
|---|---|
| 权重 | 默认本项目实验 YAML 的 `ckpt_dir`（原始 Base），与 CPT 起点相同；`--model` 可换为任意兼容的 LitGPT checkpoint |
| 权重完整性 | 参考加载器是 `strict=False`；本实现另外按 mmap 读取 checkpoint 的键并要求与模型完全一致 |
| tokenizer、任务、长度、样本 | Base 目录 tokenizer，模型 `block_size` 取实验 `context_length=32768`；完整评测按本地任务快照分 main 和 niah 两阶段，NIAH 仅使用 32768；沿用同一 harness 的任务定义、seed、生成参数和 `--limit` |
| 生成与停止 | 共用本地快照的 `litgpt.generate.base.generate`，同截断与 EOS 规则（共享代码不执行 `until` 字符串） |
| 选择题打分 | 参考代码一次前向 context+continuation 并用 `lm_head_start` 标出 context 末位；ChunkKV 以此为 prompt 边界：先 prefill 并压缩 context，再让 continuation 读压缩缓存，等价于逐 token 解码 |
| rolling PPL | 每个窗口只有 1 个 context token，整窗一次 dense 前向，ChunkKV 不改变其结果 |
| KV 统计 | 请求结束、reset 前快照。所选块计为 `exact_slots`，观察窗与之后追加的 token 计为 `recent_slots`，`hierarchical_slots=0`；比例为 Σretained/Σdense |
| 缓存预算记录 | `cache_budget` 写明预算来源、`prompt_budget_tokens`、每 token 字节数、参考 LogKV 字节数；ChunkKV 缓存按请求动态分配，`persistent_cache_bytes` 为 null |
| 计时与显存 | 沿用参考 prefill/decode 计时边界；prefill 时间包含打分与压缩 |
| 来源记录 | `comparison_contract` 含 Base/tokenizer/评测代码/兼容层哈希与依赖版本，另含实际评测权重的 SHA-256（按参考规则解析 `step_*` 后的目录）及 dense 对照的 manifest |

## 预算口径

ChunkKV 是固定预算方法（与 N 无关），SemanticLogKV 是 `O(log N)`。比较时至少报告一种对齐预算：

| 选项 | 含义 | 默认实验下的数值 |
|---|---|---|
| `budget: logkv`（默认） | 先用实验 YAML 的 LogKV 参数构建一次参考缓存，取其已分配持久 buffer 字节数（与 SemanticLogKV 自己 `swa_window_size=0` 的 SWA 对照同一口径），除以 ChunkKV 每 token 的 K/V 字节数 | K=12、B=128、recent=2048、Alpha 256 时为 2,037,252,224 字节 → **17,763 条/层/KV head（32K 的 54%）**，CPU 上用真实 1.7B 配置算得，以实际运行打印为准 |
| `--budget-from-result <logkv结果.json>` | 用 SemanticLogKV 结果中 `generate_until` 请求结束时的平均保留条目数（每层每 KV head）作为 prompt 预算，即逻辑条目数对齐 | 取决于实测；生成 token 另外追加 |
| `--budget N` | 固定条目数 | — |
| `--keep-ratio r` | 每条 prompt 保留 `ceil(r·T)`，论文常用 10% | — |

字节对齐给 ChunkKV 的预算相当宽松（参考 LogKV 的持久分配包含空槽、recent 双缓冲、原始 K 副本等）；
条目对齐更接近论文 "同 KV 条目数" 的比较。建议两者都报，并用 `compare_results` 输出的实际保留比例核对。

## 可选：同数据 dense CPT 对照（训练接口）

ChunkKV 不需要训练，主表行应为 "Base + ChunkKV"。SemanticLogKV 是在 Base 上做 32K prolong CPT 后的
模型，为了区分 "CPT 语料/预算" 与 "压缩注意力" 的贡献，提供一个对照训练入口：

```bash
# 训练环境：兼容的 PyTorch/CUDA 环境，训练依赖由 extra 安装
cd /data/lihourun/baselineForLogKV/chunkKV
python -m pip install -e '.[chunkkv-train]'
python -m chunkkv.train --print-config
NPROC_PER_NODE=8 bash scripts/chunkkv_train.sh configs/chunkkv/dense_cpt_qwen1.7b_32k.yaml
bash scripts/chunkkv_evaluate.sh --model outputs/dense_cpt/qwen1.7b-32k-attach-alpha-beta_k12_b128_2k
```

它私有加载本项目的 `demo.py` 快照，使用同一份自有实验 YAML（Base `ckpt_dir`、数据、batch、步数、优化器、
学习率日程、FSDP 与 checkpoint 格式）；仅把模型类换成子类，使 `enable_log_kv_training` 保持标准
causal attention。固定改动只有：`run_eval: ""`、`save_ckpt: true`、`resume_dir: null`、
`log_kv_semantic_clusters: false`（避免安装路由重放），输出写到本仓库 `outputs/`，并拒绝写入
本地运行快照、原始 Base 或配置中 SemanticLogKV 的 save/tensorboard/staging 目录。`--prepare` 只在首节点运行一次，记录解析后的
配置、YAML 链、训练代码与 Base 权重哈希；之后 torchrun 每个 rank 重新解析并核对未变化才训练。
`overrides` 中改动数据或预算的键会进入 manifest，但会使对照不再等价。

## 运行

完整入口 `scripts/chunkkv_evaluate_all.sh` 对齐当前 `eval.sh` / `majob.sh` 的默认任务列表：

- `main`：14 个通用任务入口、20 个 LongBench、13 个 LongBench-E，共 47 个入口。`mmlu` 等任务组会由 harness 进一步展开。
- `niah`：`niah_single_1,niah_single_2,niah_single_3`，仅评测 32768 tokens。

两个阶段各启动一次 torchrun，输出分别进入 `--output` 根目录下的 `main/` 和 `niah/`；
main 阶段失败时立即停止，不继续启动 NIAH。任务名与顺序保持和本地 Semantic 源码快照一致；NIAH 长度按本次实验要求固定为 32K。
LongBench 等真实数据任务使用 32K 模型上下文上限，保留原始样本，不把较短样本补齐到 32K。
安装 `[eval]` 会包含 [lm-eval 0.4.9 的 ruler、longbench、ifeval 依赖](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/pyproject.toml)。

```bash
# 评测环境：lm-eval==0.4.9、datasets<4；不安装相邻项目
cd /data/lihourun/baselineForLogKV/chunkKV
python -m pip install -e '.[eval]'

# 只解析本项目配置，不加载权重或训练/评测依赖
python -m chunkkv.evaluate --suite main --print-config
python -m chunkkv.evaluate --suite niah --print-config
# 各任务先用 4 条样本检查完整评测流程；默认加载原始 Base
bash scripts/chunkkv_evaluate_all.sh --limit 4 --output outputs/cmp-chunkkv
# 同一套任务评测 SemanticLogKV CPT 权重，便于核对实际输入
bash scripts/chunkkv_evaluate_all.sh --backend logkv --model /path/to/semanticlogkv/ckpt --limit 4 --output outputs/cmp-logkv
# 以双方 main 阶段本次生成的真实文件名替换 TIMESTAMP；niah 阶段同样核对
python -m chunkkv.compare_results outputs/cmp-chunkkv/main/eval_results_TIMESTAMP.json outputs/cmp-logkv/main/eval_results_TIMESTAMP.json

# 全量评测
bash scripts/chunkkv_evaluate_all.sh --output outputs/chunkkv-full
# 如只需 LongBench/LongBench-E，选择同一配置中的 33 个任务
bash scripts/chunkkv_evaluate.sh --suite longbench --output outputs/chunkkv-longbench
```

`NPROC_PER_NODE` 控制每卡一份模型。删除 `--limit` 才是任务全量；`--tasks` 为各方法选同一组任务。
`chunkkv_evaluate.sh` 不带 `--suite` 或 `--tasks` 时仍保留旧的两项 32K NIAH 小检查；
完整评测请使用新的 `chunkkv_evaluate_all.sh`。`--suite` 与 `--tasks` 互斥。
主任务阶段不传 NIAH metadata，与 Semantic 默认脚本一致。NIAH 阶段统一使用原始 Base tokenizer
作为 `metadata.pretrained`；Semantic 原脚本使用其 checkpoint 路径作为这一缓存标识，所以旧结果不能
仅凭任务名相同就认定样本一致，应核对逐样本 prompt/target 或通过上面的统一入口重跑所需对照。
层间复用：`--reuse-layers 2`。在同一 SemanticLogKV CPT 权重上换成 ChunkKV（`--model` 指向它）只是
缓存机制诊断，不是各方法各自适配后的公平比较。

### 训练平台作业：majob.sh

项目根目录的 `majob.sh` 是纯评测入口：激活环境 → main → 32K NIAH，默认加载原始
Qwen3-1.7B-Base，不执行 CPT。它只调用本项目的 `chunkkv.evaluate`，无需部署 SemanticLogKV。
默认激活 `/home/ma-user/anaconda3/bin/activate torch218`；环境名称可通过
`CHUNKKV_CONDA_ENV` 修改，或设置 `CHUNKKV_PYTHON=/absolute/path/to/python` 直接使用已准备的解释器。

提交前在目标镜像/环境中一次性安装评测依赖，确保 Python 3.10–3.13、PyTorch >= 2.7
且 CUDA 可用。环境名 `torch218` 不代表实际版本。不要同时安装 DMS 的 `[train]` extra，
它要求 `datasets==4.4.2`，与此处评测的 `datasets<4` 冲突；有冲突时使用单独的评测环境。

```bash
source /home/ma-user/anaconda3/bin/activate torch218
cd /data/lihourun/baselineForLogKV/chunkKV
python -m pip install -e '.[eval]'
```

训练平台的启动命令设为以下内容，**每个节点执行一次**，由脚本启动该节点的 GPU 进程。
先用 `--limit 4` 检查路径、缓存和分布式流程，通过后删除该参数进行全量评测。

```bash
cd /data/lihourun/baselineForLogKV/chunkKV
bash majob.sh configs/chunkkv/experiment_qwen1.7b_32k.yaml \
  --limit 4 --output outputs/chunkkv-base-32k-smoke

# 全量作业的启动命令
bash majob.sh configs/chunkkv/experiment_qwen1.7b_32k.yaml \
  --output outputs/chunkkv-base-32k-full
```

| 平台变量 | 用途 |
|---|---|
| `MA_NUM_GPUS` | 每节点 GPU 进程数；缺省依次使用 `NPROC_PER_NODE`、`LOCAL_WORLD_SIZE`、8 |
| `MA_NUM_HOSTS` | 节点数；缺省使用 `GROUP_WORLD_SIZE` 或 1 |
| `VC_TASK_INDEX` | 节点 rank；缺省使用 `NODE_RANK`、`GROUP_RANK` 或 0 |
| `MA_VJ_NAME`、`MA_TASK_NAME`、`MA_MASTER_INDEX` | 与 Semantic 原入口相同的主节点 DNS；无作业名时使用 `MASTER_ADDR` |
| `MASTER_PORT` | 默认 6000；main 默认使用其 +1、NIAH 使用 +2，可由 `EVAL_MASTER_PORT`、`NIAH_MASTER_PORT` 分别覆盖 |

所有节点需能读取同一路径下的代码、Base checkpoint、完整评测缓存，并使用同一输出路径。
无需额外共享文件 barrier：每个阶段的 torchrun rendezvous 负责节点汇合；main 失败则不进入 NIAH。
默认缓存与权重路径沿用前文的 `/home/ma-user/...`，缓存需提前准备齐全（评测默认离线）。
输出分别位于指定目录的 `main/`、`niah/`。普通单机终端仍可使用 `scripts/chunkkv_evaluate_all.sh`。

该入口为 CUDA GPU 作业设计；每张卡持有完整模型、分担评测样本。增加 GPU 数能分摊任务，
不会降低单条 32K 输入的单卡显存需求。LongBench 的 32K 指上下文上限，NIAH 只生成/评测 32768 长度。

### HellaSwag 预处理的 Arrow schema 错误

若栈为 `hellaswag.process_docs → dataset.map → _map_single → Dataset.from_file`，
并报 `Tried reading schema message, was null or length 0`，失败点是转换结果的 Arrow 文件回读，
尚未开始该任务的模型推理。不能仅凭此认定原始 HellaSwag 内容或 ChunkKV 模型接口有问题；
多个 rank 共用转换缓存文件、共享挂载的文件可见性异常是需要排查的原因。

评测入口现在通过 `datasets.disable_caching()` 让自动生成的 map/filter 缓存使用各进程独立的
临时文件，退出后恢复原来的开关。原始 `load_dataset()` 离线缓存仍然使用，不清理或重新下载数据。
这是 [Hugging Face 的转换缓存机制](https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.disable_caching)，
会增加每次启动的 CPU 预处理时间和临时磁盘占用。

平台上可显式使用节点本地临时盘，避免临时文件也写回桶挂载：

```bash
cd /data/lihourun/baselineForLogKV/chunkKV
TMPDIR=/tmp bash majob.sh configs/chunkkv/experiment_qwen1.7b_32k.yaml \
  --limit 4 --output outputs/chunkkv-base-32k-smoke
```

`--limit 4` 限制评测样本数，不保证只预处理四条数据。修复不掩盖 Arrow 异常；
若仍报错，应继续定位实际文件和挂载状态，而不是清空整套离线缓存。

### 离线缓存报 `Feature type 'List' not found` / `'Json' not found`

若栈为 `ConfigurableTask.download → datasets.load_dataset → packaged_modules/cache/cache.py →
DatasetInfo.from_directory`，失败点是读取离线缓存条目的 `dataset_info.json`，而非缓存路径找不到；
此时尚未开始该任务的模型推理，也没有进入 ChunkKV 代码。栈中嵌套两层任务组的是 `mmlu`
（`mmlu → mmlu_humanities 等 → 各学科`），对应缓存目录 `cais___mmlu/<学科>/`；排在它前面的 boolq、
hellaswag、arc、openbookqa 等已正常读取。

离线模式下，`datasets` 对每个数据集配置选用目录修改时间最新的 `<config>/<version>/<hash>` 条目。
特征类型 `List` 只由 `datasets>=4` 写出；从 datasets 4.7 起，键不统一的 dict 列（如 IFEval 的
`kwargs`）还会存成 `List(Json())`。评测环境的 datasets 3.x（traceback 行号与 3.5.1/3.6.0 一致）
不认识这些类型。也就是说，某个使用新版 datasets 的作业在共享缓存里重新准备了这些数据集，新条目盖过了
此前可读的条目。离线的 `datasets>=4` 读取已有条目时不会写新条目，所以写入者通常是联网（或经镜像）
重新准备数据的作业，也可能是从别处拷入的缓存。

这并非 ChunkKV 与 Semantic 评测方式不同所致。数据仍由本地快照中同一个 `eval.main → simple_evaluate`
加载，缓存根目录、离线变量和 `load_dataset` 包装都与 Semantic `eval.py` 相同。本入口只有两处不同：
`disable_caching()` 和把工作目录切到输出目录。两者只决定 map/filter 结果放在哪里、相对路径怎样解析，
不参与缓存条目的选择。用 datasets 3.6.0 复现时，两种方式在同一缓存状态下报同样的错误，移走新条目后
都能恢复。按代码推断，在 torch218 中用 Semantic 自己的 `eval.py` 只跑 `mmlu`、`--limit 1`，
也会得到同一错误，可用来交叉确认。

曾尝试在评测作用域内把 `List` 元数据翻译成旧类型（提交 3144672）：mmlu 能读了，但加载到 IFEval 时改报
`Feature type 'Json' not found`。`Json` 在 datasets 3.x 中没有等价类型：4.7 把这些 dict 存成 JSON
字符串、读取时再解码，3.x 则存成包含全部键的 struct，缺失键为 `None`，无法无损翻译。该补丁还让本入口
读取 Semantic 原脚本读不了的条目，数据来源与 Semantic 不同。因此已移除，评测路径恢复为与 Semantic
`eval.py` 一致，改为修复缓存本身，使双方读取同一批旧条目。

评测入口遇到这一错误时会改报 `RuntimeError`，指向下面的检查工具，原始 traceback 保留在 `__cause__` 中。

检查和处理如下。报告模式不改动缓存；`--quarantine` 只移动、不删除，需要时可原样移回。
请在没有其他作业读取该缓存时执行：

```bash
cd /data/lihourun/baselineForLogKV/chunkKV
source /home/ma-user/anaconda3/bin/activate torch218
python -m chunkkv.check_dataset_cache
python -m chunkkv.check_dataset_cache \
  --quarantine /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/unreadable_by_datasets3
```

报告一次列出所有"最新条目不可读"的数据集配置（不论是 `List`、`Json` 还是其他新类型），以及该条目的
修改时间（可据此找到写入它的作业）和隔离后将重新被选中的可读条目。隔离后，离线加载回到这些可读条目，
通常也就是此前 Semantic 评测读取的版本；新版 datasets 同样能读这些旧条目。

报告为"no readable entry"的配置没有旧条目可退回，Semantic 原脚本同样读不了（例如 2026-09-23 写入的
`truthfulqa___truthful_qa/generation`、`multiple_choice`，main 的 `truthfulqa_gen/mc1/mc2` 依赖它们）。
报告会逐个注明能否离线转换。`List` 只是 datasets 4 对列表的新写法：`List(x)` 与 3.x 的 `Sequence(x)`、
`List({...})` 与 3.x 的 `[{...}]` 对应完全相同的 Arrow 类型，所以可以不联网、不改数据地转换：

```bash
python -m chunkkv.check_dataset_cache \
  --quarantine /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/unreadable_by_datasets3 --convert
```

`--convert` 先把这些配置的条目（连同其他不可读条目）移入隔离目录，再在原路径写出一份副本：Arrow
数据逐批原样复制，只把 `dataset_info.json` 与 Arrow schema 里的特征描述改成 3.x 写法；`cache-*.arrow`
（map/filter 结果）不复制，需要时自动重算。写入前核对翻译后的特征与 Arrow 物理类型一致，写入后核对
datasets 能按该特征读回且每个 batch 与原文件相同；原文件、副本哈希与特征记录在隔离目录的
`conversions.json`。datasets 4.4.2/5.1.0 读取副本得到与原条目相同的特征和行。`Json`（4.7+，Arrow
扩展类型）、定长的 struct 列表等 3.x 无对应写法的类型不转换，仍需在评测环境中联网或经镜像对该数据集
执行一次 `load_dataset` 重新生成旧格式条目。报告中与评测任务无关的条目（例如 `ucinlp___drop`、本地
json/parquet 构建器缓存）可以不处理。
不要为此升级评测环境的 datasets，lm-eval 0.4.9 + datasets<4 是双方共同的评测口径。今后使用新版
datasets 的作业应设置独立的 `HF_DATASETS_CACHE`，否则还会写入新的条目。

## 验证记录

2026-10-15 离线缓存新版特征类型：用 datasets 3.6.0（服务器 traceback 的行号与可用类型列表都与之一致）、
4.4.2 和 4.7.0 复现。4.x 写出的 `cais___mmlu/abstract_algebra` 条目较新时，参考方式（转换缓存开启、默认
工作目录）和 ChunkKV 方式（`disable_caching()`、输出目录）都报 `Feature type 'List' not found`。
4.7.0 由同一 JSONL 准备的 IFEval 条目为 `kwargs: List(Json())`，3.6.0 为全部键的 struct 列表。加上
3144672 的补丁后 mmlu 可读、IFEval 报 `Feature type 'Json' not found`，与服务器现象一致。
`check_dataset_cache --quarantine` 移走两个新条目后，不带任何补丁的 3.6.0 读回旧条目，4.7.0 也能读这些旧条目；
离线的 4.4.2 读取已有 3.x 条目时不写新文件。`tests/chunkkv/test_check_dataset_cache.py` 与原转换缓存
回归在 datasets 3.6.0 和 4.4.2 下各 **2 passed**；Linux、Python 3.13、datasets 3.6.0 下，轻量配置/启动器
测试连同这两项共 **37 passed，2 deselected**。未接触服务器上的真实缓存，也未执行真实模型任务。

2026-10-15 Arrow 转换缓存隔离：轻量配置/启动器测试 **35 passed，2 deselected**。
额外使用 datasets 3.6.0、PyArrow 17.0.0 与三个合成样本执行真实 Arrow 回归 **1 passed**：
两个独立进程通过实际评测入口绕过空的共享转换缓存，输出相同，临时目录不同，原始文件哈希不变。
这验证缓存隔离行为，不代表已复现或验证服务器的八卡共享挂载故障；未执行真实模型任务。

2026-10-15 训练平台入口：配置、启动器与隔离测试 **29 passed，2 deselected**（命令同下）。
通过模拟解释器检查单节点/双节点的 rank、world size、主节点 DNS、独立端口与首阶段失败停止；
多节点误用 localhost 会在启动前报错。Bash 语法与 diff 空白检查通过；未验证真实 CUDA/NCCL 作业。

2026-10-15 完整任务对齐：配置、启动器与隔离测试 **26 passed，2 deselected**（命令同下）。
新增检查覆盖三种后端的任务与长度一致性、main 阶段不传 NIAH metadata、捞针长度上界、
两个独立启动阶段及首阶段失败即停止，以及各 suite 在禁止模型/外部项目导入时的配置预览。
任务清单与 Semantic 当前 `eval.sh` / `majob.sh` 做了静态一致性核对；未执行真实 GPU 全量任务。

2026-10-15 独立化改动：在 macOS、Python 3.11 下执行配置与隔离测试，结果为 **20 passed，2 deselected**：

```bash
python -m pytest -q tests/chunkkv/test_standalone.py tests/chunkkv/test_evaluate.py tests/chunkkv/test_train.py \
  -k 'not dense_control_gpt and not private_trainer'
```

覆盖本地 YAML 与环境变量展开、相对路径、输出保护、训练 manifest 和续训检查、三种后端的参数传递，
以及从其他工作目录启动、禁止外部项目与模型库导入时的配置预览。额外核对：平铺配置的 57 个字段与
原配置链展开结果一致；运行快照的语法、内部导入闭包与来源哈希一致；运行快照及许可证纳入打包文件；
两个启动脚本的 Bash 语法通过。完整小模型测试未运行：临时环境的 PyTorch 下载过慢，已停止安装；
这两项被排除的模型/训练器测试和其余模型测试不计入本次通过结果。CUDA 与真实权重验证同样未执行。

以下是 2026-10-13 的历史验证，发生在项目独立化之前，不能作为本次改动的通过记录。
当时环境：Linux CPU，Python 3.13、PyTorch 2.9.1、lightning 2.6.6、lm-eval 0.4.9、datasets 3.6.0、
transformers 4.56.2、litdata 0.2.59；SemanticLogKV 参考为 `HourunLi/LogKV@4fd97c7`，editable 安装核心包。

```bash
python -m pytest -q tests/chunkkv
```

历史结果：**34 passed**。覆盖：

- 打分与选块和逐 head、逐 query 的显式 softmax/循环实现一致；窗口保留、尾块截断、比例预算、
  稳定排序、短 prompt 与参数校验。
- 不压缩时，ChunkKV 包装与 LitGPT 原生 dense KV cache 的 prefill/decode logits 一致（float32，1e-5）。
- 压缩后 decode 与 "屏蔽被丢弃 prompt 位置" 的原生 dense 推理一致；每层用自己的 Q/K 打分、
  复用层沿用上一打分层索引，缓存内容等于 dense K 在所选位置的取值。
- 选择题拆分路径等于 "context prefill + 逐 token 解码"；缓存扩容、输入位置与生命周期检查。
- 通过真实私有加载的 `eval.py`：参考构造函数加载 tiny checkpoint 与 tokenizer，不压缩时
  `generate_until`、`loglikelihood`、`loglikelihood_rolling` 与参考 dense 后端一致；压缩时统计字段正确；
  `logkv` 预算等于参考 LogKV 缓存的持久字节换算；缺键 checkpoint 被拒绝；参考文件与原生类不变。
- 启动器：三种后端共享任务输入，只有 LogKV 或字节对齐时传 LogKV 参数；输出隔离与结果重标注；
  条目对齐预算读取；结果核对拒绝 prompt/运行时不同。实际执行 `--print-config` 四种组合成功。
- dense 对照：解析真实实验 YAML 链后训练超参不变；输出与输入保护；manifest 与续训一致性；
  子类在 demo.py 的调用后 loss 与原生 GPT 一致；私有加载真实 `demo.py` 不改参考文件。

尚未验证：CUDA 上的真实 1.7B 权重推理、RULER 实际数据与得分、多卡评测、显存与速度、
dense 对照的真实训练（`demo.py` 需要 CUDA）。先在目标机器用 `--limit 4` 跑通三种后端并
`compare_results` 核对，再扩大任务。
