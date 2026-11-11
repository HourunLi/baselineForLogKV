# SnapKV baseline（Qwen3-1.7B-Base，SemanticLogKV 评测口径）

本分支是一个**独立项目**：在 Qwen3 上实现 SnapKV，并按 SemanticLogKV 的数据集、请求接口和指标口径评测，作为 LogKV 的横向对比 baseline。
目前只接入 32K 正式评测。

- SnapKV 的选择规则移植自 [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)（MIT，`68cd955`）的 `SnapKVCluster`；
  该实现沿用 SnapKV 原作者仓库 [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)（Apache-2.0，`e216ddc`）的
  `SnapKVCluster.update_kv`。许可证见 [LICENSE](LICENSE) 与 [licenses/SnapKV-Apache-2.0.txt](licenses/SnapKV-Apache-2.0.txt)。
- 评测侧的模型、tokenizer、KV cache、生成、打分、指标和结果格式全部在本仓库实现，与 `PyramidKV`、`H2O` 分支是同一套代码
  （取自 `PyramidKV` 分支 97b4337，即加入 64K/128K 之前的 32K 版本，只改了方法相关部分）。**不 import、不加载 litgpt / SemanticLogKV 的任何代码**，
  只依赖 PyTorch、tokenizers、lm-eval 等通用库。
- 与 SemanticLogKV 对齐的内容是“行为”：同一份原始 Base 权重、同一 tokenizer 规则、同一 lm-eval 任务与缓存、同样的截断/生成/打分规则、
  同样的 KV 槽位与速度指标定义、同样的结果文件布局。
- 统一实现规则见 [AGENTS.md](AGENTS.md) 与 [Baseline 实现规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md)，评测环境、路径、依赖写法与已知坑见
  [Baseline 环境与避坑清单](docs/BASELINE_ENVIRONMENT.md)。正式评测只用 32K 窗口，完整任务为项目内
  [configs/eval_suites.yaml](configs/eval_suites.yaml) 维护的 47 个 main 入口和 3 个 NIAH（仅 `[32768]`），由 `majob.sh` 分两阶段运行。

## 为什么不直接跑上游代码

SnapKV 原仓库与 KVCache-Factory 都通过 monkey-patch HF 的 Llama/Mistral attention 接入（分别锁定 transformers 4.37 与 4.44.2，
没有 Qwen3 的 q/k norm），各自的 LongBench/NIAH runner 使用自己的数据、prompt 模板、HF `generate` 停止规则和打分，与 SemanticLogKV 不同。
因此本分支只移植 SnapKV 的算法核心（打分、池化、top-k 保留），再放进与 SemanticLogKV 同一接口的评测器中。

## 方法实现

每层的流程与上游 `SnapKVCluster.update_kv` 相同：

1. **Prefill**：用整段 prompt 的完整 K/V 做因果注意力（prefill 输出不受压缩影响）。
2. **评分（观察窗）**：最后 `window_size` 个 prompt query 对全部 key 做 softmax（窗口内因果），对更早的 key 在窗口上求和；
   `kv_head` 布局下按 GQA 组聚合，再做 1-D pooling（`kernel_size`）。
3. **统一预算**：**每层都**保留 top `max_capacity_prompt - window_size` 个旧 token 加最近 `window_size` 个 token，共 `max_capacity_prompt` 条。
   prompt 短于 `max_capacity_prompt` 时整段保留（与上游的 `q_len < max_capacity_prompt` 判断相同）。
4. **真实删除**：每个 head 只存保留的 K/V（post-RoPE），其余不再存储。
5. **Decode / 续写**：新 token 追加到压缩后的 cache，不再淘汰（上游语义：只在 prefill 压缩一次）。

| 参数 | 默认值 | 来源 |
|---|---|---|
| `max_capacity_prompt` | 2048 | SnapKV 原仓库 `init_snapkv` 默认；论文 LongBench 报告 1024/2048/4096 |
| `window_size` / `kernel_size` / `pooling` | 8 / 7 / `maxpool` | KVCache-Factory `run_longbench.py` 对 `snapkv` 的设置（与 PyramidKV、H2O 分支相同） |
| `kv_cache_granularity` | `kv_head` | 每个 KV head 一份条目，与 SemanticLogKV 的 KV group 槽位同单位；`query_head` 复现原仓库布局（K/V 先展开到每个 query head，GQA 下条目翻倍） |
| `gqa_score_agg` | `mean` | KVCache-Factory `kv_head` 路径的组内聚合，`max`/`sum` 可选 |

与上游的差异及影响：

- **窗口长度**：SnapKV 论文的 LongBench 配置为 `window 32 / kernel 7 / maxpool`（原仓库 `experiments/LongBench/config/*_w32_k7_maxpool.json`）。
  这里默认用 KVCache-Factory runner 的 `window 8`，使 SnapKV、PyramidKV、H2O 三个分支在同一预算下只有选择规则不同；需要论文设置时传
  `--window-size 32`（保留总条目不变，被选的旧 token 少 24 个）。两种设置的结果须分别标注，不能混在一起比较。
- **GQA**：原仓库针对 MHA（或先 `repeat_kv`）的张量；Qwen3-1.7B 是 16 个 query head / 8 个 KV head，默认 `kv_head` 先在组内聚合分数，
  再按 KV head 选 token，存储量与 dense GQA cache 逐条对应。`kv_head` 的打分等价于 KVCache-Factory 先 `repeat_kv` 再聚合的写法，只是不实际复制张量。
- **存储顺序**：上游按分数顺序 gather，这里按位置排序后存储；attention 与条目顺序无关，保留集合和输出相同（测试逐元素核对）。
- **与 PyramidKV 的关系**：两者打分与选择代码相同，区别只在逐层预算——PyramidKV 低层多、高层少（各层均值约为预算），SnapKV 各层相同。
  同一 `max_capacity_prompt` 下两者的保留条目总数基本一致，可直接对照。

**选择题/PPL 打分**（`--loglikelihood-mode`）：SnapKV 只在一次前向结束后压缩，所以同一次前向内的 token 看到的都是完整 KV。
默认 `compress_context` 先 prefill 并压缩 context，再在压缩后的 cache 上为 continuation 打分（与生成答案时的处境一致）；
`single_forward` 一次前向打完分，结果等同 FullKV。context 短于预算时两者相同；rolling PPL 的 context 只有 1 个 token，不会触发压缩。
NIAH 是生成任务，不受此选项影响。与 PyramidKV、H2O 分支的默认值一致。

## 与 SemanticLogKV 的对齐

| 项目 | 本分支做法 |
|---|---|
| 权重 | 直接读取 SemanticLogKV 实验 `ckpt_dir` 下的原始 `lit_model.pth`（参数名按 LitGPT 布局，无需转换），也支持 HF safetensors；严格加载，缺/多张量即报错。免训练，直接在原始 Base 上评测 |
| tokenizer | 自行实现与参考评测相同的规则：BOS/EOS 来自 `tokenizer_config.json` 与 `generation_config.json`，自动去除后端多加的 BOS/EOS，不加 chat template |
| 任务与数据 | 评测环境中的 lm-eval（torch218 为 0.4.11；任务定义随版本变化，见“环境”一节）；`--suite main` 为参照版本 `eval.sh` / `majob.sh` 的 47 个 main 入口（14 通用、20 LongBench、13 LongBench-E，任务组由 lm-eval 展开），不传 NIAH metadata；`--suite niah` 为 `niah_single_1/2/3`，`metadata.max_seq_lengths=[32768]`，`metadata.pretrained` 原样使用 Base 路径字符串；HF/RULER 离线缓存根目录相同，评测强制离线，不提供联网下载开关 |
| 截断 | 打分时左截断 context；生成时先预留 `max_gen_toks` 再左截断 prompt；空 context 用 `[bos]` |
| 生成 | 非采样时 temperature/top_p 置 0 走 argmax；只在 EOS 或长度上限停止，不执行 `until` 字符串（与参考评测一致） |
| 打分 | continuation `bos=False` 单独编码；fp32 log-softmax，1024 位置分块；rolling 窗口 `max_seq_length-1` |
| KV 指标 | 请求结束、释放 cache 前快照；分母为已前向 token × 层 × KV group，不含最后一个未回填的输出 token；汇总用 ΣS/ΣD；`recent`=窗口+追加 token，`exact`=选中的旧 token，`hierarchical`=0 |
| 速度/显存 | 同样的 prefill（含首 token 采样，SnapKV 压缩计入 prefill）/decode 计时边界与每请求 CUDA 峰值 |
| 多卡 | 每卡一份模型，按 `requests[rank::world]` 分配，Gloo 汇总后按原顺序重排 |
| 预处理缓存 | 原始离线 `load_dataset` 缓存照常读取；map/filter 结果用 `datasets.disable_caching()` 写进程私有临时文件，多 rank 不竞争共享挂载上的同一缓存文件 |
| 输出 | `eval_results_<run_id>.json`（`cache_budget`、`kv_compression`、`inference_metrics`、`metadata`、`results`，含 lm-eval 逐样本日志）及 `.kv_compression.csv`、`.inference_metrics.xlsx`、`.csv`、`resolved_eval_<run_id>.yaml`；`run_id` 为启动时间，同目录已存在时自动加后缀，不覆盖历史结果 |

`cache_budget.persistent_cache_bytes` 为 `null`：cache 按需增长（prefill 后为预算，decode 逐 token 追加），没有 SemanticLogKV 那样的预分配常驻缓冲；
逐请求的 `retained_kv_bytes` 给出保留条目的 K/V 字节数（不含预留空位与临时打分工作区）。

## 环境

评测在 SemanticLogKV 的评测环境 torch218 中进行，**不安装本项目，也不往其中安装任何包**（包括 `[eval]` extra 和 `pip install -e .`）。
入口脚本先 `cd` 到项目根目录再用 `python -m` 启动，torchrun 的各个进程也从这里 import `snapkv`，所以只需要把代码放到服务器上：

```bash
# 首次检出：仓库的默认分支是 DMS，必须用 -b 指定本分支；已有目录不能靠 git pull 换分支
git clone -b snapKV https://github.com/HourunLi/baselineForLogKV.git /data/lihourun/baselineForLogKV/snapKV
source /home/ma-user/anaconda3/bin/activate torch218
cd /data/lihourun/baselineForLogKV/snapKV
git branch --show-current    # 应为 snapKV；目录里应有 snapkv/，不应有 dms/ 或其他方法的包
```

2026-10-16 的 torch218 为 torch 2.11.0+cu128、lm-eval 0.4.11、datasets 4.8.4、transformers 4.51.3、tokenizers 0.21.4、pyarrow 23.0.1，
完整清单见 [docs/BASELINE_ENVIRONMENT.md](docs/BASELINE_ENVIRONMENT.md) 第 1 节。代码适配这个环境：

- 评测入口在 import 评测框架之前只检查代码和共享缓存的最低要求：`lm-eval>=0.4.9`（本项目用到的 `LM` 请求接口、`simple_evaluate`、`TaskManager` 与 `make_table`）、
  `datasets>=4.7`（共享缓存中 IFEval 条目的 `Json` 特征类型从 4.7 起可读）。不满足时报错，提示回到 torch218 评测，而不是改动环境。
- 精确版本写进每个结果的 `provenance.versions`（torch、transformers、datasets、pyarrow、lm-eval、tokenizers、numpy），另有解释器、CUDA 运行时与驱动、
  各 rank 的主机与 GPU。`compare_results` 要求双方这六个包的版本一致。
- `pyproject.toml` 只写代码真正 import 的包的下限，torch218 全部满足，也不排除其中任何版本。`[eval]`（`lm-eval[ruler,longbench,ifeval]>=0.4.9`、
  `datasets>=4.7`、`pandas>=1.5`、`xlsxwriter>=3`）只用于另建环境；transformers 随 lm-eval 而来，不单独声明。prefill 使用 PyTorch SDPA，
  不需要 FlashAttention、accelerate、sentencepiece 或 openpyxl。

**lm-eval 版本决定任务定义。** 0.4.9 与 0.4.11 的 LongBench / LongBench-E 数据源（`THUDM/LongBench` → `Xnhyacinth/LongBench`）、
prompt 字段（`{{input}}` → `{{question}}`）和解码方式（`do_sample: True` → `False`）都不同；本项目生成时 `do_sample` 为假即 argmax，
所以 0.4.11 下 LongBench 是贪心解码。arc、ceval、ifeval 与 RULER（NIAH）任务文件两版无差异。只比较同一 lm-eval 版本下的结果；
SemanticLogKV 来自其他版本的旧结果须在 torch218 中重跑。详见 [docs/BASELINE_ENVIRONMENT.md](docs/BASELINE_ENVIRONMENT.md) 第 5 节。

## 运行方式

以下命令都在仓库根目录执行。

每次运行必须二选一：`--suite main|niah|longbench`（项目内任务清单，输出写入 `<output>/<suite>/`）或 `--tasks`（诊断用的显式任务列表）。

```bash
# 0) 只解析配置，不加载权重和数据（不 import torch / lm_eval）
python -m snapkv.evaluate --config configs/qwen3_1.7b_32k_snapkv.yaml --suite main --print-config
python -m snapkv.evaluate --config configs/qwen3_1.7b_32k_snapkv.yaml --suite niah --print-config

# 1) 每个任务 4 条样本的冒烟检查（单卡）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_snapkv.yaml --suite niah --limit 4
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_snapkv.yaml --suite main --limit 4

# 2) 同一流程的 FullKV 对照（不删 KV，检查流水线本身）
bash scripts/evaluate.sh configs/qwen3_1.7b_32k_snapkv.yaml --suite niah --method fullkv --limit 4

# 3) 预算扫描（可选，不属于默认流程），结果分别写到 outputs/snapkv-b<预算>/<suite>/
BUDGETS="128 512 2048" bash scripts/sweep_budget.sh configs/qwen3_1.7b_32k_snapkv.yaml --suite niah --limit 50
```

常用覆盖项：`--max-capacity-prompt`、`--window-size`、`--kernel-size`、`--pooling`、`--kv-cache-granularity`、`--gqa-score-agg`、`--limit`、`--output`、
`--model`（SemanticLogKV 的 `ckpt_dir` 或 HF 目录）、`--hf-cache-base`（已准备好的离线缓存根目录）。评测从不联网下载数据；缺数据时直接报错。
共享离线缓存只读：报 `Feature type 'List'/'Json' not found` 说明安装的 datasets 比写缓存的版本旧（环境被降级），应回到 torch218，而不是转换缓存；
某个条目的 `dataset_info.json` 不是合法 JSON 时报错并停止，不移动、不改写、不删除缓存条目，处理方法见
[docs/BASELINE_ENVIRONMENT.md](docs/BASELINE_ENVIRONMENT.md) 第 4 节。

### 训练平台作业：majob.sh（完整评测）

`majob.sh` 是正式完整评测入口，**每个节点执行一次**：先 main（47 个入口），再 NIAH（3 个入口，仅 32768）。两个阶段各自一次
`torch.distributed.run`、各用一个端口，输出分别在 `<output>/main/` 与 `<output>/niah/`；main 失败时返回非零并且不启动 NIAH。
它不允许 `--suite/--tasks/--eval-suites/--include-path/--ruler-lengths/--ruler-pretrained/--max-seq-length` 等会改变协议的参数，
只接受方法、预算、输出目录与调试用 `--limit`。

```bash
cd /data/lihourun/baselineForLogKV/snapKV    # 不安装任何包；默认激活 torch218
bash majob.sh configs/qwen3_1.7b_32k_snapkv.yaml --limit 4 --output outputs/snapkv-32k-smoke   # 小样本检查
bash majob.sh configs/qwen3_1.7b_32k_snapkv.yaml --output outputs/snapkv-32k-full              # 全量
```

| 平台变量 | 用途 |
|---|---|
| `MA_NUM_GPUS` | 每节点 GPU 进程数；缺省依次使用 `NPROC_PER_NODE`、`LOCAL_WORLD_SIZE`、本节点可见的 CUDA 设备数；超过可见设备数时在启动 torchrun 前报错 |
| `MA_NUM_HOSTS` | 节点数；缺省使用 `GROUP_WORLD_SIZE` 或 1 |
| `VC_TASK_INDEX` | 节点 rank；缺省使用 `NODE_RANK`、`GROUP_RANK` 或 0 |
| `MA_VJ_NAME`、`MA_TASK_NAME`、`MA_MASTER_INDEX` | 主节点 DNS（`<作业>-<任务>-<序号>.<作业>`）；无作业名时使用 `MASTER_ADDR`，多节点禁止 localhost |
| `MASTER_PORT` | 只用于多节点：默认 6000，main 用 +1、NIAH 用 +2，可由 `EVAL_MASTER_PORT`、`NIAH_MASTER_PORT` 覆盖。单节点每个阶段用 `--standalone`，由 torchrun 绑定空闲端口 |
| `SNAPKV_PYTHON` / `SNAPKV_CONDA_ENV` | 直接指定解释器；否则激活 `/home/ma-user/anaconda3/bin/activate ${SNAPKV_CONDA_ENV:-torch218}`。环境名不代表实际版本，以结果中的 `provenance` 为准 |

激活环境后、任何 CUDA 调用之前，脚本照搬 SemanticLogKV `eval.sh`/`majob.sh` 的运行库设置：把 `/usr/local/cuda-12.8/bin` 加到 `PATH` 前面，
`/usr/lib/x86_64-linux-gnu` 与 `/usr/local/cuda-12.8/lib64` 加到 `LD_LIBRARY_PATH` 前面，并设置同样的 `CUDA_DEVICE_MAX_CONNECTIONS`、`NCCL_*` 变量
（训练作业默认的 `LD_LIBRARY_PATH` 会先加载旧 libcuda，报 `driver ... too old (found version 11040)`）。随后用同一解释器执行一次 `torch.cuda.init()`：
失败时打印加载的 libcuda 路径、驱动 API 版本和 `LD_LIBRARY_PATH` 后退出，不启动 torchrun。`scripts/evaluate.sh` 使用同样的库路径。

脚本导出 `KV_BASELINE_EXPECTED_WORLD_SIZE=节点数×每节点 GPU`，评测入口发现实际进程数不同时报错。所有节点需读到同一路径的代码、Base、离线缓存，并写同一输出目录。
平台上建议设置 `TMPDIR=/tmp`，让进程内 map/filter 的临时文件落在节点本地盘。

### 与 SemanticLogKV 横向对比

1. 在 SemanticLogKV 自己的项目里生成同一组样本的结果（以 NIAH 为例：单档 32K，`pretrained` 用同一个 Base 路径；main 阶段按其默认任务列表、不传 metadata）。
   本项目不读取、不调用 SemanticLogKV 的代码或配置，下面的命令在对方项目中执行：

   ```bash
   DIAG_ARGS='--limit 4 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
     bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_1,niah_single_2,niah_single_3 none
   ```

2. 选定预算口径（三选一，结果里分别标注）：
   - **默认 2048 条/层/KV head**：KVCache-Factory 与 SnapKV 原仓库的默认值，与 PyramidKV 分支默认预算相同。
   - **按条目对齐**：取该 SemanticLogKV 结果在生成请求结束时的平均保留槽位数（`--budget-from-result`，按条目数，不是字节）：

     ```bash
     bash scripts/evaluate.sh configs/qwen3_1.7b_32k_snapkv.yaml --suite niah --limit 4 \
       --budget-from-result /path/to/semanticlogkv/eval_results_TIMESTAMP.json --output outputs/snapkv-matched
     ```

   - **按持久分配字节对齐**：默认 LogKV 配置（K=12、B=128、recent=2048、Alpha 256）的持久分配为 2,037,252,224 字节
     （[环境清单](docs/BASELINE_ENVIRONMENT.md) 第 5 节，以实际运行打印为准）。SnapKV 每条 K+V 为 `2 × 128 × 2` 字节（bf16），
     28 层 × 8 个 KV head 均匀分配时为每层每 KV head 17,763 条（17,763 × 114,688 = 2,037,202,944 字节），即 `--max-capacity-prompt 17763`。
     这是 prefill 后的保留量，decode 再逐 token 追加（最多 `max_gen_toks`），且 prompt 短于预算时不压缩。

3. 核对评测协议与实际测试输入一致，并并排输出分数、KV 保留率、速度和显存：

   ```bash
   python -m snapkv.compare_results /path/to/semanticlogkv/eval_results_TIMESTAMP.json \
     outputs/snapkv-matched/niah/eval_results_RUNID.json outputs/fullkv-full/niah/eval_results_RUNID.json
   ```

   协议检查分两层：逐样本 doc/prompt/target 哈希与请求参数（含生成参数）必须一致；lm-eval 记录的 `limit`、各任务样本数、任务版本与任务设置
   （输出类型、fewshot、生成参数、分隔符）必须一致；本项目结果额外记录的窗口长度、tokenizer 哈希和 torch/transformers/datasets/pyarrow/lm-eval/tokenizers
   版本若双方都有则必须一致，SemanticLogKV 自身结果缺这些字段时列为“未核对”。任一不一致直接报错。方法字段（后端、预算、权重哈希）单独列出，
   不参与判定。PyramidKV、H2O 分支的结果文件也可以一起传入。

`--method fullkv` 与 SemanticLogKV 的 `log_kv_dense_mode: true` 应给出几乎相同的 NIAH 分数；两者使用不同的 attention 实现，逐 token logits 可能有 bf16 级别差异。

## 输出

`--output` 目录（默认 `outputs/<method>-b<预算>`；使用 `--suite` 时再进入 `<suite>/` 子目录）中：

| 文件 | 内容 |
|---|---|
| `resolved_eval_<run_id>.yaml` | 本次运行解析后的全部设置（方法参数、任务、窗口、缓存目录、32K prompt 的逐层条目数） |
| `eval_results_<run_id>.json` | 与 SemanticLogKV 相同的主字段，另含 `run_id`、`suite`、`model_backend`、`method_config`、`provenance`（权重/tokenizer/代码 SHA-256、git 提交、依赖版本、解释器、CUDA/驱动、各 rank 主机与 GPU） |
| `eval_results_<run_id>.kv_compression.csv` | 每次模型调用的 KV 槽位快照 |
| `eval_results_<run_id>.inference_metrics.xlsx` | 速度汇总、逐请求明细、按 prompt 长度分桶（需 pandas 与 xlsxwriter，torch218 自带） |
| `eval_results_<run_id>.csv` | 各任务的全部数值指标（NIAH 的指标名为长度，如 `32768,none`） |

KV 指标分三类分别报告：逻辑 KV 条目（`kv_compression`，ΣS/ΣD）、保留条目的 K/V 字节（逐请求 `retained_kv_bytes`，不含预留空位与打分工作区）、
实际 CUDA 峰值显存（`inference_metrics` 的 `peak_memory_*`）。三者不能互相替代。

## 测试

CPU 即可运行，使用随机小模型，不下载任何权重或数据。在 torch218 或其副本中、项目根目录下运行，不设 `PYTHONPATH`、不安装本项目
（副本的建法见 [docs/BASELINE_ENVIRONMENT.md](docs/BASELINE_ENVIRONMENT.md) 第 1 节；副本只能证明 Python 层兼容）：

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

覆盖：SnapKV 保留的条目与上游代码逐元素一致（`query_head` 对照 FasterDecoding/SnapKV 的 `update_kv` 原文，`kv_head` 对照 KVCache-Factory 的
`_grouped_window_attn_cache`/`_select_topk_kv` 原文，两种 pooling、三种组内聚合）；各层预算相同、短 prompt 整段保留；以 Transformers
`Qwen3ForCausalLM` prefill + 上游 SnapKV 逐层压缩 + 在真实位置 decode 为参照，本项目模型的 prefill/decode logits 一致；真实删除后的条目计数；
Qwen3 与 Transformers 参考实现数值一致；LitGPT/HF 两种权重布局加载一致；带 cache 的分段 prefill/续写/逐 token decode 与整段前向一致；
请求截断、BOS、EOS、rolling 窗口、按 rank 分配；任务清单（47 main、33 LongBench、3 NIAH 仅 32768）、main 阶段不传 NIAH metadata、
`--print-config` 在禁止 import torch/lm_eval/litgpt 等的情况下只读项目内 YAML、启动检查放行 torch218 的版本并拒绝过旧版本、`pyproject` 只有
torch218 满足的下限、`majob.sh` 的可见卡数默认值、CUDA 自检与库路径、单节点 `--standalone`、多节点的节点/端口/world size 解析与首阶段失败即停止、
协议字段不一致时拒绝比较、运行编号不覆盖历史结果；以及一次经 lm-eval 的端到端运行（生成 + 选择题，SnapKV 与 FullKV 结果通过输入一致性校验）。
验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 注意事项

- 这是免训练方法，直接在原始 Base 上评测；SemanticLogKV 使用 CPT 后的权重，比较时需注明权重不同。
- SnapKV 的 prefill 本身是完整注意力，峰值显存包含整段 prompt 的单层 K/V 与打分工作区；它节省的是 prefill 之后常驻的 cache 和 decode 访存。
- 默认预算 2048 下，prompt 短于 2048 token 的任务（多数通用 main 任务）不压缩，结果应与 FullKV 相同；压缩主要作用于 LongBench 与 NIAH。
- 槽位比例不等于显存比例；`query_head` 布局下保留率可能大于 1（每个 KV group 存多份）。
- 尚未在真实 1.7B 权重、真实离线缓存和 CUDA/多节点作业上运行，也没有全量结果；请先用 `majob.sh ... --limit 4` 确认端到端链路，再去掉 `--limit` 做全量。
