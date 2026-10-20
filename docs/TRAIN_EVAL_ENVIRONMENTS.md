# 服务器训练与评测环境规范

本文是 [Baseline 实现规范](BASELINE_IMPLEMENTATION_GUIDE.md) 的环境与运行补充，区分同一服务器上的训练 Python 环境和评测 Python 环境。二者都可运行在 GPU 作业平台上；“训练平台”不表示所有作业都需要训练。

下文的依赖组、`BASELINE_*` 变量、配置文件名和 CLI 是**新 baseline 应提供的通用接口约定**，不代表已有项目均已支持。`your_baseline`、`cuda-base` 及示例配置必须由项目实现者替换或落实；命令是使用模板，本规范本身不创建环境、安装依赖或启动作业。

## 1. 两个环境各负责什么

| 维度 | 训练环境 | 评测环境 |
|---|---|---|
| 职责 | 必要的数据准备、训练、续训和模型导出 | 加载完成的模型产物、任务数据，运行统一评测 |
| 是否必须 | 仅需训练或转换的算法使用；免训练方法跳过 | 所有正式 baseline 必须具备 |
| 依赖 | 方法所需的训练框架、数据处理与导出依赖，独立锁版本 | 统一评测框架与任务依赖；现有 torch218：`lm-eval==0.4.11`、`datasets==4.8.4`；完整当前取值以 [环境清单](BASELINE_ENVIRONMENT.md) 为准，运行记录精确版本 |
| 数据 | 固定训练语料与该训练环境生成的缓存 | 经验证的评测数据、tokenizer、RULER 等资产 |
| 写入 | 本方法训练缓存、日志、checkpoint 与导出目录 | 本次结果目录、节点本地临时变换文件；不改变正式数据内容 |
| 分布式 | 依据方法选择训练策略并记录 | 遵循统一协议分配样本、汇总结果；不以训练分布式设置替代评测设置 |
| 验证 | 数据、损失、梯度、保存/恢复与导出格式 | 完整任务覆盖、实际输入、评分、生成、指标与结果可追溯性 |

两个环境可以读取同一份原始 Base 和 tokenizer。评测只接收已完成的权重/导出产物，不读取训练环境的 Python 模块，也不靠激活训练环境才能反序列化模型。若产物含方法特有结构，所需模型加载代码必须已经在本项目评测依赖中声明。

## 2. 环境如何建立和选择

评测直接使用现有 torch218，禁止在其中安装本项目或变更包。训练可从一个已验证 Python/PyTorch/CUDA 兼容的基础环境建立独立环境。环境名称只是标识，不能据此推断版本；不要求把所有方法的训练依赖锁成同一组。

项目先维护 `requirements/train-constraints.txt` 和 `requirements/eval-constraints.txt`，分别固定实际 PyTorch、数据处理及其他相关依赖的精确版本。首次建立时按已验证的 GPU 基础环境和各阶段依赖要求填写，在隔离环境验证后固定完整结果；不能直接把训练约束复制给评测。镜像版本、CUDA/PyTorch 包来源也要记录，版本号清单不能替代这些信息。

以下在项目根目录执行，首次准备时运行一次。`cuda-base` 是待替换的基础环境名，constraints 文件须已在本项目准备好；已有合格的两套环境时直接使用，不重复创建。[Conda 支持克隆环境](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#cloning-an-environment)。

```bash
set -e
conda create --name baseline-train --clone cuda-base

source /home/ma-user/anaconda3/bin/activate baseline-train
python -m pip install -c requirements/train-constraints.txt -e '.[train]'
python -m pip check

# 评测仅选择已有环境，不运行 pip install / pip check 修复共享平台包
source /home/ma-user/anaconda3/bin/activate torch218
```

需要训练的项目分别声明 `train`、`eval` 依赖组；免训练项目只需评测环境及其依赖，不必提供空训练入口。不能要求安装 `[train,eval]` 才能评测。安装应服从已验证的精确版本约束，不能默许依赖解析器替换已验证的 CUDA/PyTorch 组合。安装和模型/数据准备在作业提交前完成，正式作业不现场升级包。

两个环境分别保存实际 Python 路径、Python/PyTorch/CUDA 版本、完整依赖清单和相关内核检查结果。`pip check` 只检查声明的依赖关系，不证明 GPU 内核或真实任务可运行。开发阶段可在任一环境修改同一项目源码；正式训练/评测期间固定代码版本，不在运行中的 editable checkout 上改代码。

启动器必须支持显式解释器。单阶段使用 `BASELINE_PYTHON`；一体化入口分别使用 `BASELINE_TRAIN_PYTHON` / `BASELINE_EVAL_PYTHON`，再回落到 `BASELINE_PYTHON`。HybridCSA 的训练只用 torch218 已有的 PyTorch、LitData、PyArrow，因此不另建训练环境，训练与评测都使用 torch218 的同一解释器；其方法变量、默认值与优先级见 [运行说明](HYBRID_CSA.md#51-平台一体化入口)。不得无提示重新激活某个固定环境。分布式启动统一用 `"$BASELINE_PYTHON" -m torch.distributed.run`，避免 `python` 与 PATH 中的 `torchrun` 来自不同环境。

## 3. 环境变量与优先级

每个作业显式加载对应环境配置，不能依赖前一次交互终端留下的变量。**切换 conda 环境不会自动切换 HF 缓存或项目路径。**

### 项目变量

| 变量 | 约定 |
|---|---|
| `BASELINE_PROJECT_ROOT` | 当前项目根的绝对路径；用于 shell 定位。入口仍从自身位置确定所属项目，不能用它把代码/配置重定向到外部项目 |
| `BASELINE_PYTHON` | 本次训练或评测所用解释器的绝对路径 |
| `BASELINE_BASE_CHECKPOINT` | 原始 Base 资产目录；实际评测的训练产物通过显式 `--model` 指定 |
| `BASELINE_DATA_DIR` | 本方法训练数据资产目录；免训练评测不要求设置 |
| `BASELINE_OUTPUT_ROOT` | 本次运行输出根目录，训练和评测分别命名；不得指向权重或缓存资产目录 |

参数优先级统一为：**显式 CLI > 对应环境变量 > 项目 YAML > 项目默认值**。只对声明支持的字段进行映射；项目必须说明映射关系，不能假定任意环境变量会自动影响配置。

HF 子目录已被显式设置时，优先使用对应变量，不能再被 `HF_HOME` 的派生默认路径静默覆盖。最终选中的解释器、数据/缓存和输出路径必须在启动日志中显示，并写入解析配置。

### 数据与运行变量

| 变量 | 训练环境 | 评测环境 |
|---|---|---|
| `HF_HOME` | 本方法训练缓存根 | 固定评测缓存根 |
| `HF_DATASETS_CACHE` | 本训练环境专用的可写 datasets 缓存 | 经验证的评测 prepared dataset cache |
| `HF_HUB_CACHE` | 训练用 Hub 缓存 | 评测用 Hub 缓存 |
| `HUGGINGFACE_HUB_CACHE` | 依赖需要旧变量时，设成与 `HF_HUB_CACHE` 一致 | 同左，不得指向另一个数据版本 |
| `HF_MODULES_CACHE` | 训练数据加载模块缓存（如需要） | 对应评测环境的固定模块缓存（如需要） |
| `HF_EVALUATE_CACHE` | 用到时显式设置，不复用评测的可写目录 | 对应协议的 evaluate 资源缓存 |
| `RULER_CACHE_DIR`、`NLTK_DATA` | 仅在训练需要时设置 | 指向已准备的任务资源 |
| `HF_HUB_OFFLINE`、`HF_DATASETS_OFFLINE` | 正式训练使用已准备数据时设为 `1`；联网准备另起明确的准备作业 | 必须为 `1` |
| `TMPDIR` | 节点本地可写临时目录 | 节点本地可写临时目录，变换文件按进程隔离 |
| `TOKENIZERS_PARALLELISM` | 明确设置；本规范默认 `false` | `false`，避免各 rank 隐式扩展 tokenizer 线程 |
| `HF_DATASETS_IN_MEMORY_MAX_SIZE` | 依据训练方式设置并记录 | 默认 `0`，避免把完整缓存隐式预载入内存；不等于数据处理没有内存占用 |

Hub 下载缓存和 datasets 的 Arrow 缓存是不同目录，必须分别确认；只设置其中一个不能控制另一个。[datasets 缓存说明](https://huggingface.co/docs/datasets/v3.6.0/cache)。这些变量应在相关库首次导入前设置；在导入后修改 shell 配置不会改变已经运行的进程。[Hub 环境变量说明](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。

训练与评测不得共享同一个**可写**数据准备目录。评测资产可以由多个 baseline 读取，但数据内容与版本固定；库需要的锁文件写权限应单独考虑，不能把普通“内容只读”规则误当成任意目录都能物理只读挂载。预处理变换写入各进程临时目录，不能污染 prepared dataset cache。

仅设置 `TMPDIR=/tmp` 不会把默认的 `Dataset.map/filter` 写入自动移走。评测入口还必须显式禁用 datasets 的持久变换缓存，或给变换指定各进程独占路径；仅设置 `load_from_cache_file=False` 也不等于隔离写入。此设置只针对变换结果，仍需读取原始离线数据缓存。

`HF_DATASETS_TRUST_REMOTE_CODE`、`HF_ALLOW_CODE_EVAL` 等只在固定任务确实需要时由项目明确启用并记录；不能为了省事在用户的全局 shell 中开启。`CUDA_VISIBLE_DEVICES` 沿用平台分配，作业不得自行暴露未分配的 GPU。通信/分配器调优变量按目标环境验证后设置，不把某个历史作业的取值当成所有方法的必需项。

## 4. 相对路径统一怎么解析

| 路径来源 | 基准 |
|---|---|
| 项目默认配置、脚本和运行代码 | 入口文件所在项目的根目录；与调用者当前目录无关 |
| 直接 Python CLI 传入的相对路径 | 调用者启动该命令时的工作目录，进入业务逻辑前立即解析 |
| YAML 继承或引用另一个 YAML | 写出该引用的 YAML 所在目录；解析软链接后仍不得越出项目边界 |
| YAML 中的权重、数据、缓存和输出相对路径 | 项目根目录，不相对 YAML 的层级，也不相对后续临时工作目录 |
| 环境变量中的资产/输出路径 | 要求绝对路径，不依赖当前工作目录 |
| Python 包导入 | 已安装的本项目包和声明依赖；不得靠外部项目 `PYTHONPATH` 补齐 |

shell 启动器定位并切换到自身项目根后再调用 Python，因此**通过项目 shell 脚本传入的相对 CLI 路径统一相对项目根**；直接运行 Python CLI 则遵循上表的调用目录语义。平台提交建议使用绝对项目路径，项目内配置和输出可使用明确的项目相对路径。

例如项目为 `/data/lihourun/baselineForLogKV/your_baseline`，`configs/experiment.yaml` 中引用 `base.yaml`，则父配置是 `configs/base.yaml`；其中的 `output: outputs/train/run-001` 指项目根下的 `outputs/train/run-001`。共享 checkpoint 使用绝对资产路径，不用 `../../另一个项目/...` 绕过独立性规则。

配置加载先完成显式支持的环境变量展开、`~` 展开与路径解析，再创建输出目录或切换工作目录。未设置的必需变量应报错，不能当作空字符串或保留字面量继续运行。保存实际解析路径；不能在训练环境、评测环境和不同 rank 中分别猜测路径。

## 5. 训练环境如何设置与运行

以下是设置模板。`your_baseline`、训练数据路径以及 `configs/train-smoke.yaml` 由项目实现者填写；配置预览与训练入口必须由该项目提供。免训练方法跳过本节。

```bash
set -e
source /home/ma-user/anaconda3/bin/activate baseline-train
export BASELINE_PROJECT_ROOT="/data/lihourun/baselineForLogKV/your_baseline"
export BASELINE_PYTHON="$(python -c 'import sys; print(sys.executable)')"
export BASELINE_BASE_CHECKPOINT="/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/"
export BASELINE_DATA_DIR="/absolute/path/to/prepared-training-data"
export BASELINE_OUTPUT_ROOT="$BASELINE_PROJECT_ROOT/outputs/train/run-001"

export HF_HOME="$BASELINE_PROJECT_ROOT/.cache/train/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0 TMPDIR=/tmp

cd "$BASELINE_PROJECT_ROOT"
"$BASELINE_PYTHON" -m your_baseline.train --config configs/train-smoke.yaml --print-config
# 训练数据准备和短步数配置检查通过后执行
bash scripts/train.sh configs/train-smoke.yaml
```

空的新缓存不会自动包含训练数据；首次运行前应在训练准备环境将固定语料准备到上述目录，或让训练配置明确读取已有的 prepared data 资产。确需联网的数据准备单独进行，仅写训练缓存；不得为此关闭正式评测的离线开关。短步数检查通过后再选择正式训练配置，不以检查命令触发全量训练。

训练脚本使用选定解释器，负责必要的数据准备、分布式训练和导出。共享数据准备只由一个负责进程执行；其他进程等准备成功后再读取，准备失败不能继续训练。续训必须显式选择 checkpoint，并核对配置、数据和方法参数；不能把目录中任意文件当成可恢复状态。

## 6. 评测环境如何设置与运行

重新设置解释器和缓存变量，避免继承训练环境的值。下面保留既有评测资产布局：数据集缓存是 `.../data/hf_cache/hf_cache`；这是现有目录约定，不是 HF 的通用默认。目录迁移后显式填写实际位置，不能只因名字重复而改路径。

```bash
set -e
source /home/ma-user/anaconda3/bin/activate torch218
export BASELINE_PROJECT_ROOT="/data/lihourun/baselineForLogKV/your_baseline"
export BASELINE_PYTHON="$(python -c 'import sys; print(sys.executable)')"
export BASELINE_BASE_CHECKPOINT="/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/"
export BASELINE_OUTPUT_ROOT="$BASELINE_PROJECT_ROOT/outputs/eval/run-001"

export HF_HOME="/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_EVALUATE_CACHE="$HF_HOME/evaluate"
export RULER_CACHE_DIR="$HF_HOME/ruler_cache"
export NLTK_DATA="$HF_HOME/nltk_data"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0 TMPDIR=/tmp

cd "$BASELINE_PROJECT_ROOT"
# 需训练的方法：替换为已导出并验证完成的评测权重目录
EVAL_MODEL_DIR="/absolute/path/to/completed-model-export"
# 免训练方法则使用：EVAL_MODEL_DIR="$BASELINE_BASE_CHECKPOINT"
"$BASELINE_PYTHON" -m your_baseline.evaluate --config configs/eval.yaml \
  --model "$EVAL_MODEL_DIR" --suite main --print-config
"$BASELINE_PYTHON" -m your_baseline.evaluate --config configs/eval.yaml \
  --model "$EVAL_MODEL_DIR" --suite niah --print-config

# 完成环境、缓存和必要的单进程实际读取检查后提交；平台每节点执行一次
bash majob.sh configs/eval.yaml --model "$EVAL_MODEL_DIR" \
  --limit 4 --output "$BASELINE_OUTPUT_ROOT"
```

首次先用少量真实样本验证链路；通过后使用新的输出目录、去掉 `--limit` 执行完整 32K 评测。配置预览不读取权重，缓存元数据检查不等于完整任务已通过。评测入口不得从 `BASELINE_DATA_DIR` 的训练语料推断评测数据，也不能因训练产物缺失就退回 Base。

## 7. 平台变量与启动约定

训练脚本和评测 `majob.sh` 都必须明确支持平台多节点配置；不允许训练脚本默认单机却对外宣称支持多节点。

| 平台变量 | 解析要求 |
|---|---|
| `MA_NUM_GPUS` / `NPROC_PER_NODE` | 每节点 GPU 进程数；平台值优先，不能超过已分配设备数 |
| `MA_NUM_HOSTS` / `NNODES` | 节点数；单机为 1 |
| `VC_TASK_INDEX` / `NODE_RANK` | 节点 rank；必须位于 `[0, 节点数)` |
| `MASTER_ADDR` | 所有节点可达的同一主节点地址；多节点不能使用 localhost |
| `MA_VJ_NAME`、`MA_TASK_NAME`、`MA_MASTER_INDEX` | 平台未显式提供 `MASTER_ADDR` 时，可按其约定构造主节点 DNS |
| `MASTER_PORT` | 本作业基准端口；训练可使用基准端口，main 与 NIAH 分别使用不同端口，例如 `+1`、`+2` |
| `RANK`、`LOCAL_RANK`、`WORLD_SIZE` | 由 torchrun 给子进程设置，用户不逐卡手写；总进程数应与拓扑一致 |

多节点时每节点执行一次项目脚本，脚本内部启动 torchrun；外部不要再包一层 torchrun。`--standalone` 只用于单机检查。不同并发作业必须避免 rendezvous 端口冲突。

所有节点使用一致的代码、依赖、资产身份和配置，并能访问各自所需的资产与输出。`/tmp` 是节点本地临时空间，不是跨节点共享目录；checkpoint、导出产物和汇总结果放在明确的持久路径。脚本应打印解释器与版本、节点 rank/数量、进程数、主节点地址/端口和关键路径，失败返回非零状态。

## 8. 从训练交接到评测

训练结束后，导出由项目评测加载器支持的模型格式，保存实际权重、模型结构、tokenizer 来源及训练记录。分片/优化器 checkpoint 只有在评测加载器明确支持时才可直接传入；否则先在训练环境完成导出。

必须等保存与导出完整结束、文件检查通过，再把确定的目录交给评测。评测不能读取正在写入的 checkpoint，也不能因为训练进程退出成功就假定导出成功。交接的是模型资产和记录，不是训练环境、源码路径或其依赖缓存。

最终验收包括：训练与评测解释器正确、依赖分别可复现、缓存写入隔离、相对路径基准一致、跨目录启动可用、实际导出产物能被评测环境读取、少量样本与必要分布式检查通过。环境搭建、配置预览和 shell 语法检查不能替代这些运行验证。
