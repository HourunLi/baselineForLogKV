# Baseline 环境与避坑清单

本文件补充 [AGENTS.md](../AGENTS.md) 与 [Baseline 实现规范](BASELINE_IMPLEMENTATION_GUIDE.md)：规范规定“要做到什么”，
本文件记录评测环境、路径、环境变量、依赖声明和平台入口的**当前取值**，以及 ChunkKV 接入时实际踩过的坑。
新 baseline 开工前通读一遍；环境变化后先更新第 1 节，再改代码。方法自身的细节与排错记录仍写在各自的 `docs/<METHOD>.md`。

快照日期：2026-10-16（torch218 由用户导出的完整环境清单）。未注明的条目是源码或环境事实；实际跑过的标“已验证”，推断标“推断”。

## 0. 最重要的十条

1. **只在 torch218 中评测，代码适配环境。** 不在 torch218 里 `pip install` 任何东西，包括本项目的 extra 和 `pip install -e .`。
2. **依赖只写代码真正需要的下限。** 任何约束都必须被 torch218 现有版本满足；像 `datasets<4`、`lm-eval==0.4.9` 这种排除现有版本的写法，一旦安装就会改动共享环境。
3. **启动时检查最低版本，结果里记录精确版本。** 比较工具拒绝版本不同的两份结果。
4. **lm-eval 版本决定任务定义。** 0.4.9 与 0.4.11 的 LongBench 数据源、prompt 和解码方式都不同，只比较同一版本下的结果。
5. **共享离线缓存只读。** 读不了时先查环境版本，不转换、不隔离、不删除缓存条目。
6. **不 `import litgpt`。** torch218 里装着 SemanticLogKV 的 editable litgpt；复用代码放进本项目私有命名空间。
7. **入口脚本先 `cd` 到项目根目录，再用 `python -m` 启动。** 不靠安装、`PYTHONPATH` 或软链接。
8. **GPU 数从平台变量或可见设备推出，不写死 8。**
9. **先 `--print-config`，再 `--limit 4`，最后全量。** main 与 NIAH 都要过小样本。
10. **改动先在 torch218 副本里跑测试。** 没在服务器或 GPU 上跑过的，交付时明确写“未验证”。

## 1. 评测环境 torch218

环境名不代表版本（`torch218` 实际是 torch 2.11）。以下为 2026-10-16 的环境事实。

| 类别 | 版本 |
|---|---|
| 解释器 | `/home/ma-user/anaconda3/envs/torch218/bin/python`，Python 3.10.20，EulerOS（glibc 2.31） |
| 深度学习 | torch 2.11.0+cu128、torchvision 0.26.0+cu128、triton 3.6.0、flash-attn 2.8.3 |
| 评测 | **lm-eval 0.4.11**、**datasets 4.8.4**、transformers 4.51.3、tokenizers 0.21.4、huggingface-hub 0.36.2、evaluate 0.4.6、fsspec 2023.10.0 |
| 数据与表格 | pyarrow 23.0.1、numpy 2.2.6、pandas 2.3.3、xlsxwriter 3.2.9 |
| 训练相关 | lightning 2.6.1、jsonargparse 4.41.0、litdata 0.2.61、safetensors 0.7.0、tensorboard 2.20.0 |
| 其他 | PyYAML 6.0.3、psutil 7.1.3、nltk 3.9.4、pytest 9.1.1、hf_transfer 0.1.9 |
| editable 包 | `litgpt 0.5.12`，来自 `gitee.com/yushengzhao/litgpt@3ae1ade`，即 SemanticLogKV 的 fork |
| **未安装** | accelerate、sentencepiece、openpyxl |
| 硬件 | 8× A800-SXM4-80GB，驱动 535.183.01 |

需要注意的环境事实：

- **CUDA 版本只认 `torch.version.cuda`（12.8）。** 镜像里的 `CUDA_VERSION=11.7.1`、`CUDA_PKG_VERSION` 是旧值。（推断）驱动 535 原生对应 CUDA 12.2，cu128 依靠 `LD_LIBRARY_PATH` 中的 `/usr/local/cuda-12.8/compat` 运行。需要编译 CUDA 扩展的方法，先确认 `nvcc --version` 与 `torch.version.cuda` 一致，能不编译就不编译。
- **`PYTHONPATH=/usr/local/seccomponent/lib:/home/ma-user/infer/model/1`。** 包名和顶层模块名要避开 `utils`、`data`、`model`、`litgpt` 这类通用名字。
- **`HF_ENDPOINT=https://hf-mirror.com`。** 只在联网准备数据时有用，评测强制离线。
- **`pip check` 原本就报冲突**（modelarts、msprof-analyze 对 attrs、pandas、numpy 等的约束）。这是环境原状，不要试图“修复”。
- **lm-eval 0.4.9 在 torch218 里连 import 都失败（缺 accelerate）。** 已验证（副本中）。所以“把 lm-eval 锁到 0.4.9”在这个环境里不可行。

### 导出环境快照（只读）

环境有变化或怀疑被改动时，在服务器上运行，把输出文件发出来核对：

```bash
source /home/ma-user/anaconda3/bin/activate torch218
OUT=/data/lihourun/torch218_env_$(date +%Y%m%d_%H%M%S).txt
{
  echo "## python"; which python
  python -c "import sys, platform; print(sys.version); print(platform.platform())"
  echo; echo "## key packages"
  python - <<'EOF'
import importlib.metadata as m
for n in ("lm-eval", "datasets", "transformers", "torch", "pyarrow", "tokenizers", "huggingface-hub",
          "accelerate", "lightning", "jsonargparse", "numpy", "pandas", "litgpt", "triton", "flash-attn"):
    try:
        print(n, m.version(n))
    except m.PackageNotFoundError:
        print(n, "not installed")
import torch
print("torch.cuda", torch.version.cuda, "| cuda available", torch.cuda.is_available())
EOF
  echo; echo "## pip freeze"; python -m pip freeze --all
  echo; echo "## pip check"; python -m pip check
  echo; echo "## conda list"; conda list
  echo; echo "## GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo; echo "## environment variables (tokens/keys removed)"
  env | grep -E '^(HF_|HUGGINGFACE|RULER|NLTK|PYTHONPATH|CUDA_|LD_LIBRARY_PATH)' | grep -viE 'token|key|secret|password'
} > "$OUT" 2>&1
echo "$OUT"
```

### 在开发机复现 torch218 跑测试

服务器之外改代码时，用 freeze 建一个 Python 3.10 副本跑测试。ChunkKV 的 74 项测试就是这样验证的。

```bash
# freeze.txt 来自上面的 "## pip freeze" 段。去掉平台专用包、editable litgpt、需要编译的 flash-attn 和测试用不到的
# torchvision；torch 去掉 +cu128 后缀改从 PyPI 安装，它的 nvidia/cuda 依赖由安装器重新解析
grep -vE '^(-e |modelarts|ma[-_]|tb-graph-ascend|msprof|esdk-obs|huaweicloud|obstore|boto|s3transfer|cryptography|flash.attn|pip @|nvidia-|cuda-|torchvision)' \
  freeze.txt | sed 's/+cu[0-9]*$//' > freeze-replica.txt
uv venv --python 3.10 venv-torch218
uv pip install --python venv-torch218/bin/python -r freeze-replica.txt
# 在项目根目录运行；不设 PYTHONPATH，也不安装本项目
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 venv-torch218/bin/python -m pytest -q tests
```

这样装出的 torch 2.11.0 是 PyPI 默认的 cu130 版；上述过滤与解析已在 2026-10-16 用该 freeze 验证。
副本只能证明 Python 层兼容，不能代替服务器上的 GPU、缓存和多节点验证。

## 2. 依赖声明（`pyproject.toml`）

| 规则 | 做法 |
|---|---|
| 核心依赖 | 只列代码直接 import 的包，下限不高于 torch218 中的版本。ChunkKV 的写法：`torch>=2.7`、`lightning>=2.6.1`、`jsonargparse[signatures]>=4.37,<=4.41`、`tokenizers>=0.21`、`safetensors>=0.4.3`、`PyYAML>=6.0`、`psutil>=5.9`、`tqdm>4.66` |
| `[eval]` extra | `lm-eval[ruler,longbench,ifeval]>=0.4.9`、`datasets>=4.7`、`pandas>=1.9`、`xlsxwriter>=3`。只用于另建环境，不装进 torch218 |
| 不要声明 | transformers（随 lm-eval 而来）、accelerate、sentencepiece、openpyxl。代码确需时惰性 import，并给出清楚的报错 |
| 禁止 | 排除 torch218 现有版本的约束：`lm-eval==0.4.9`、`datasets<4`、`transformers==4.57.3`，以及 torch218 没有的 `accelerate==1.4.0`，都会在安装时改动共享环境。确需上限（如 jsonargparse 的 `<=4.41`）时，必须包含 torch218 的版本 |
| 训练 extra | `[<method>-train]` 只装在单独的训练环境，例如 `litdata>=0.2.59`。训练与评测 extra 不混装 |
| 启动检查 | 在 import 评测代码之前检查 `lm-eval>=0.4.9`、`datasets>=4.7`（缓存里 IFEval 的 `Json` 类型从 4.7 起可读）。报错提示“在 SemanticLogKV 的环境中评测，不改动其中的包” |
| 版本记录 | 结果的 `comparison_contract.versions` 记录 torch、transformers、datasets、pyarrow、lm-eval、tokenizers 的精确版本；`resolved_eval` 另记解释器、CUDA、驱动、GPU 与 torchrun 拓扑 |

pandas 写 xlsx 时优先用 xlsxwriter，torch218 里只有它。在联网的副本里加载任务时，lm-eval 会自动下载 nltk 的 `punkt_tab`；
服务器离线评测时，它必须已在 `NLTK_DATA` 下。

## 3. 路径与环境变量

### 共享路径（2026-10-16）

| 用途 | 路径 | 说明 |
|---|---|---|
| 各 baseline 代码 | `/data/lihourun/baselineForLogKV/<分支名>` | chunkKV 在 `.../chunkKV`；其他分支同样每个一个目录 |
| SemanticLogKV 项目 | `/data/lihourun/semanticLogKV` | 只读参考，运行时不得访问 |
| 原始 Base | `/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/` | 保留末尾斜杠：RULER metadata 与缓存键用的就是这个写法 |
| 训练数据 prolong | `/home/ma-user/work/bucket-pangu-green/lihourun/longcontext/prolong/data` | 只供需训练的方法使用 |
| HF 缓存根 | `/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache` | datasets 条目在其中的 `hf_cache/` 子目录 |
| SemanticLogKV 输出 | `/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/...`、`tb-32k-*` | 不得写入或覆盖 |
| 旧隔离目录 | `.../data/hf_cache/unreadable_by_datasets3` | ChunkKV 排错时留下的，不要复用；内容见 `docs/CHUNKKV.md` |

`bucket-pangu-green` 是桶挂载。目录“移动”可能实际是复制加删除，修改时间不一定保留；而离线加载正是按修改时间选条目。

### 评测进程必须设置的变量

HF 库在**第一次 import 时**就读取这些变量，必须在 import `datasets`、`huggingface_hub`、`lm_eval` 之前设好。
参照 `chunkkv/evaluate.py` 的 `configure_environment()`，`<BASE>` 由 `<METHOD>_HF_CACHE` 覆盖：

```
HF_HOME=<BASE>  HF_DATASETS_CACHE=<BASE>/hf_cache  HF_EVALUATE_CACHE=<BASE>/evaluate
HF_MODULES_CACHE=<BASE>/modules  HF_HUB_CACHE=<BASE>/hub  HUGGINGFACE_HUB_CACHE=<BASE>/hub
RULER_CACHE_DIR=<BASE>/ruler_cache  NLTK_DATA=<BASE>/nltk_data
HF_HUB_OFFLINE=1  HF_DATASETS_OFFLINE=1  HF_DATASETS_IN_MEMORY_MAX_SIZE=0
HF_DATASETS_TRUST_REMOTE_CODE=1  TOKENIZERS_PARALLELISM=false  HF_ALLOW_CODE_EVAL=1
```

### 项目覆盖变量（命名约定）

| 变量 | 作用 |
|---|---|
| `<METHOD>_BASE_CHECKPOINT`、`<METHOD>_DATA_DIR` | 在实验 YAML 中写成 `${<METHOD>_BASE_CHECKPOINT-默认路径}` |
| `<METHOD>_HF_CACHE` | HF 缓存根 |
| `<METHOD>_PYTHON` / `<METHOD>_CONDA_ENV` | `majob.sh` 使用的解释器或 conda 环境名，默认 `torch218` |
| `TMPDIR=/tmp` | 平台上建议设置，让进程内 map/filter 的临时文件落在节点本地盘，不写回桶挂载 |

`majob.sh` 还会设置 `LITGPT_EXPECTED_WORLD_SIZE`（vendor 的 eval 用它核对 rendezvous）、
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 和 `NCCL_IB_TIMEOUT=22`。

## 4. 离线数据与缓存

- **离线加载怎么选条目。** 缓存目录为 `<命名空间>___<snake_case 名称>/<config>/<version>/<hash>`，例如 `Xnhyacinth___long_bench`、`google___if_eval`、`cais___mmlu`。同一 config 有多条时选修改时间最新的。
- **lm-eval 0.4.11 下 50 个入口读的数据集：** `aps/super_glue`（boolq）、`baber/piqa`、`allenai/social_i_qa`、`Rowan/hellaswag`、`allenai/winogrande`、`allenai/ai2_arc`、`allenai/openbookqa`、`cais/mmlu`（57 个 config）、`ceval/ceval-exam`（52）、`google/IFEval`、`truthful_qa`（generation）、`truthfulqa/truthful_qa`（multiple_choice）、`Xnhyacinth/LongBench`（33）。NIAH 由 RULER 用 Base tokenizer 和 `ruler_cache` 现场生成。缺失时报 `... is not cached`。
- **`Feature type 'List'/'Json' not found` 的含义。** 安装的 datasets 比写缓存的版本旧：4.x 写 `List`，4.7 起写 `Json`。这说明环境被降级了，不是缓存坏了。正确做法是回到 torch218，而不是改缓存。
- **不得修改共享缓存。** 确需处理时：只移动、不删除，移出缓存目录并记录来源与哈希，提供原样恢复的命令，执行前确认没有其他作业在用。
- **进程内预处理缓存。** 评测期间调用 `datasets.disable_caching()`，map/filter 改用各进程自己的临时文件。否则多个 rank 共用转换缓存，可能报 `Tried reading schema message, was null or length 0`（HellaSwag 遇到过）。原始 `load_dataset` 缓存照常使用。
- **NIAH 样本的身份。** `metadata.pretrained` 统一用原始 Base tokenizer 路径。SemanticLogKV 原脚本用的是其 checkpoint 路径，因此任务名相同不代表样本相同，要核对逐样本 prompt。

## 5. 评测协议的落点

- **任务清单。** 47 个 main 入口（14 个通用 + 20 个 LongBench + 13 个 LongBench-E）和 3 个 NIAH（`niah_single_1/2/3`，仅 `[32768]`），保存在项目内 YAML，可从 `configs/chunkkv/eval_semantic_default.yaml` 复制。
- **lm-eval 0.4.9 与 0.4.11 的任务定义差异**（逐文件比较两版 wheel）：
  - LongBench 数据源 `THUDM/LongBench` → `Xnhyacinth/LongBench`；
  - prompt 中问题字段 `{{input}}` → `{{question}}`；
  - `do_sample: True` → `False`；
  - 多个数据集仓库名加了命名空间；
  - arc、ceval、ifeval、RULER 无差异。
  
  vendor 生成在 `do_sample` 为假时走 argmax，因此 0.4.11 下 LongBench 是贪心解码。详见 `docs/CHUNKKV.md`“lm-eval 版本与任务定义”。
- **请求语义。** 新方法继承 vendor 中的 `LogKVLM`，只替换注意力或缓存，`loglikelihood`、`loglikelihood_rolling`、`generate_until` 全部沿用。共享生成代码按 EOS 与 `max_gen_toks` 停止，不执行 `until` 字符串。选择题用 `lm_head_start` 标出 context 与 continuation 的边界。
- **对照。** 同一入口提供 `--backend logkv`（SemanticLogKV 本身）和 `--backend dense`（Full KV），结果带 `comparison_contract`，用 `compare_results` 核对逐样本输入与生成参数。SemanticLogKV 的旧结果若来自其他 lm-eval 版本，须在 torch218 中重跑。
- **预算参照。** 默认 LogKV 配置（K=12、B=128、recent=2048、Alpha 256）的持久分配为 2,037,252,224 字节；ChunkKV 换算为每层每 KV head 17,763 条，约 32K 的 54%。这是 CPU 上用真实 1.7B 配置算得的，以实际运行打印为准。逻辑条目、分配字节和峰值显存分开报告。

## 6. 项目结构与入口约定

```
<method>/                    # Python 包；名字避开 litgpt/utils/data/model 等通用顶层名
  _vendor/semanticlogkv/     # 复用代码：SOURCE.json（来源 commit、逐文件 SHA-256、修改说明）+ LICENSE.md
  evaluate.py                # 评测 CLI
  compare_results.py
configs/<method>/            # 平铺的实验 YAML（不跨项目继承）、方法参数、任务清单
scripts/<method>_evaluate.sh, <method>_evaluate_all.sh, <method>_train.sh（如需训练）
majob.sh                     # 平台入口
tests/<method>/
docs/<METHOD>.md             # 方法细节、预算口径、排错与验证记录
```

评测 CLI 至少支持以下参数：

- 选择任务（必须显式给出，二者互斥）：`--suite main|niah|longbench` 或 `--tasks`；
- `--print-config`：只解析配置，不 import torch 或 lm_eval；
- `--limit`、`--output`、`--model`、`--backend <method>|logkv|dense`；
- 方法参数与预算参数。

每次启动写 `resolved_eval_<launch_id>.yaml`，结果文件为 `eval_results_<时间>.json`，不覆盖同目录的旧记录。

`majob.sh` 约定（参照本分支实现）：

| 项目 | 约定 |
|---|---|
| 每节点进程数 | `MA_NUM_GPUS` → `NPROC_PER_NODE` → `LOCAL_WORLD_SIZE` → `torch.cuda.device_count()`；超过可见设备数时在启动 torchrun 前报错 |
| 节点数 / rank | `MA_NUM_HOSTS`（缺省 `GROUP_WORLD_SIZE`、1）；`VC_TASK_INDEX`（缺省 `NODE_RANK`、`GROUP_RANK`、0） |
| 主节点 | 有 `MA_VJ_NAME` 时为 `${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}`，否则用 `MASTER_ADDR`；多节点禁止 localhost |
| 端口 | `MASTER_PORT` 默认 6000，main 用 +1、NIAH 用 +2，可由 `EVAL_MASTER_PORT`、`NIAH_MASTER_PORT` 覆盖 |
| 阶段 | 先 main 后 NIAH，各起一次 torchrun；main 失败即退出，不进入 NIAH |
| 拒绝的参数 | `--suite`、`--tasks`、`--evaluation-config`、`--experiment-config`、`--print-config` |
| 环境 | 默认 `source /home/ma-user/anaconda3/bin/activate torch218`；conda 激活后再 `set -u` |

## 7. 最低测试与验证顺序

建议从 ChunkKV 复制并改名的测试：

- **配置预览独立。** `--print-config` 在禁止 import `litgpt/utils/data/torch/lightning/lm_eval/litdata` 的情况下能运行，且只读项目内的 YAML。
- **`majob.sh`。** 用假的 python 核对：rank、端口、主节点地址的计算；默认取可见卡数；超出可见卡数时拒绝；main 失败后停止；拒绝协议参数。
- **环境检查。** 参数化覆盖 torch218 的实际版本（必须放行）和过旧版本（必须拒绝），并检查 `pyproject` 的 extra 与启动检查一致、没有声明 transformers。
- **对照与结果。** `compare_results` 能发现 prompt 或生成参数不同，但不因模型回答不同而报错；输出不写进参考项目。
- **vendor 副作用。** SemanticLogKV 的 `demo.py` 在 import 时执行 `torch.set_default_dtype(torch.bfloat16)`，测试里要保存并恢复默认 dtype。

验证顺序：

1. torch218 副本中全部测试通过；
2. 服务器上 `--print-config`；
3. `majob.sh --limit 4`，main 与 NIAH 都跑；
4. 同样 `--limit 4` 跑 `--backend logkv` 并 `compare_results`；
5. 全量。

交付时逐项写明哪一步在哪个环境验证过。

## 8. 踩坑记录（ChunkKV，2026-10-13 至 10-16）

| 现象 | 原因 | 正确做法 |
|---|---|---|
| 离线加载报 `Feature type 'List' not found` | 旧 `[eval]` 写了 `datasets>=2.18,<4`，在 torch218 中安装后 datasets 被从 4.8.4 降到 3.x | 不在 torch218 安装任何包；依赖只写下限 |
| 为了让 3.x 读缓存，隔离并转换了共享缓存条目 | 把环境问题当成缓存问题处理 | 缓存只读；已改动的用工具原样恢复（`check_dataset_cache --restore`） |
| 启动器拒绝 torch218 | 启动检查写成 `lm-eval==0.4.9`，而 torch218 是 0.4.11 | 检查最低版本，精确版本写进结果 |
| 安装 extra 会顺带改动其他包（静态检查发现，未实际发生） | 其他分支声明了 `transformers==4.57.3`、`accelerate==1.4.0`，torch218 是 4.51.3、未装 accelerate | 不声明代码不直接 import 的包 |
| GPU 少于 8 张的开发容器启动失败 | `majob.sh` 在没有平台变量时默认 8 个进程，多出的 rank 在 `set_device` 失败 | 默认取可见设备数，超出时提前报错 |
| 文档要求 `pip install --no-deps -e .` | 实际不需要：入口先 cd 到项目根再 `python -m` | 不安装，直接从项目根目录运行 |
| HellaSwag 报 Arrow schema 错误 | （推断）多个 rank 共用 map 转换缓存，且缓存在共享挂载上 | `datasets.disable_caching()`，并设 `TMPDIR=/tmp`；仍报错时查实际文件与挂载，不清缓存 |
| 0.4.9 与 0.4.11 分数不可比 | LongBench 数据源、字段和解码方式随 lm-eval 版本改变 | 双方在同一环境同一版本下重跑 |
| NIAH 样本不一致 | 参考脚本用 checkpoint 路径作 `metadata.pretrained` | 统一用 Base tokenizer 路径，核对逐样本 prompt |

## 9. 现有 baseline 分支的已知差距（2026-10-16 静态检查）

只读检查了 `origin/CaM`、`origin/DMS`、`origin/H2O`、`origin/PyramidKV`，未修改，也未在服务器上运行。

| 问题 | CaM | DMS | H2O | PyramidKV |
|---|---|---|---|---|
| 启动检查要求 `lm-eval==0.4.9` 且 `datasets<4`，在 torch218 中会直接拒绝运行 | 是 | 是 | 是 | 是 |
| `[eval]` 写 `datasets>=2.18,<4`、`lm-eval==0.4.9` | 是 | 是 | 是 | 是 |
| 还锁了 transformers 等包 | `==4.57.3` | `==4.57.3`、`accelerate==1.4.0` | `<5` | `<5` |
| README 要求 `pip install -e '.[eval,...]'`，会改动 torch218 | 是 | 是 | 是 | 是 |
| `majob.sh` 默认 8 个进程 | 是 | 是 | 是 | 是 |
| AGENTS.md 第 7 条和规范 4.1 仍是 `datasets<4` 的旧版 | 是 | 是 | 是 | 是 |

这些分支在 torch218 中运行前需要按 `chunkKV` 分支的做法对齐，对应提交：17e99d3、7de21cb、218eadb、81a8919、5ea70c6。
另外，各分支的 AGENTS.md、实现规范和本文件应保持同一版本。
