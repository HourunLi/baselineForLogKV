# 验证记录

只记录本分支实际执行过的检查。配置检查、CPU 局部测试与真实模型/全量评测分开表述。

## 2026-10-19：`--print-config` 独立性测试不再依赖机器上是否有 Base

PyramidKV 在服务器 torch218 上同一测试 5 项失败（用户运行）；本分支的测试相同，（推断）在服务器上会同样失败，本分支尚未在服务器上运行。失败的都是 `tests/test_standalone.py` 的 `--print-config` 测试：
服务器上真实 Base 存在，预览按设计读取其 `model_config.yaml` 取层数（共享模型元数据，规范允许），而测试只允许读项目内的 YAML；
开发容器里没有这个路径，所以此前没有暴露。评测代码未改。

修复：测试自带一个只有 LitGPT `model_config.yaml` 和空 `lit_model.pth` 的临时 Base 并用 `--model` 指向它，只额外放行这一个文件，
并断言逐层预算按其层数算出。torch218 副本中，在服务器同路径放一个同样的假 Base（复现出相同的 5 项失败）和不放两种情况下，
`python -m pytest -q` 均为 **75 passed**。修复后尚未在服务器上重跑。

## 2026-10-16：适配 torch218（lm-eval 0.4.11、datasets 4.8.4），不改动共享环境

按 [docs/BASELINE_ENVIRONMENT.md](BASELINE_ENVIRONMENT.md) 与 `chunkKV` 分支的做法对齐：启动检查只要求 `lm-eval>=0.4.9`、
`datasets>=4.7`；`pyproject.toml` 只写下限，去掉 `lm-eval==0.4.9`、`datasets<4`、`transformers<5`、openpyxl；不安装本项目；
`majob.sh` 默认取可见卡数、设置 SemanticLogKV 的 CUDA 12.8 库路径并在 torchrun 前做 CUDA 自检、单节点用 `--standalone`；
`--print-config` 不再 import torch；`compare_results` 核对 torch、transformers、datasets、pyarrow、lm-eval、tokenizers 六个版本。
H2O 的算法、请求语义、打分与结果格式未改。

环境：开发容器（CPU）中的 torch218 Python 副本：Python 3.10.20、torch 2.11.0（PyPI 默认 CUDA 版，只在 CPU 上执行）、
lm-eval 0.4.11、datasets 4.8.4、transformers 4.51.3、tokenizers 0.21.4、huggingface-hub 0.36.2、evaluate 0.4.6、fsspec 2023.10.0、
pyarrow 23.0.1、numpy 2.2.6、pandas 2.3.3、xlsxwriter 3.2.9、safetensors 0.7.0、PyYAML 6.0.3、nltk 3.9.4、pytest 9.1.1；
与 torch218 一样没有 accelerate、openpyxl、sentencepiece。副本按环境清单第 1 节的版本表逐个安装（没有完整 freeze），
只能证明 Python 层兼容。

| 检查 | 结果 |
|---|---|
| 改动前的 `origin/H2O`（1727dfd）在副本中 `python -m pytest -q` | **54 passed、1 failed**：端到端测试在启动检查处报 `requires lm-eval==0.4.9, found 0.4.11`，即在 torch218 中会拒绝运行 |
| 改动后 `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q` | **75 passed**，无 skip；含与 transformers 4.51.3 `Qwen3ForCausalLM` 的数值对照，以及经 lm-eval 0.4.11 `simple_evaluate` 的端到端运行（生成 + 选择题，H2O、H2O+decode 淘汰与 FullKV 通过输入一致性校验） |
| 任务名解析：`configs/eval_suites.yaml` 的 47 个 main 与 3 个 NIAH 交给 lm-eval 0.4.11 `TaskManager` | 50 个全部可解析，`mmlu`、`ceval-valid` 为任务组；LongBench 任务文件为 `Xnhyacinth/LongBench` 与 `{{question}}`。只检查注册与任务文件，没有读取数据 |
| `--print-config` 不依赖 torch | 在只有 PyYAML、没有 torch 的 Python 3.13 中 `--suite niah --print-config` 正常输出；改动前同一命令在 import `torch.distributed` 时失败。测试另在禁止 import torch/lm_eval/datasets/transformers/tokenizers/litgpt 等的情况下运行，并检查只读项目内 YAML |
| `scripts/evaluate.sh configs/qwen3_1.7b_32k_h2o.yaml --suite niah --print-config`（副本解释器，从项目外目录调用） | 经 `python -m torch.distributed.run --standalone` 正常输出 |
| `majob.sh` 模拟解释器测试 | 无平台变量时取可见卡数；请求数超过可见卡数、没有可见设备或 CUDA 初始化失败时在 torchrun 前退出；`LD_LIBRARY_PATH` 以 `/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64` 开头；单节点用 `--standalone`、不传固定端口；多节点的 rank、主节点 DNS、6001/6002 端口；main 失败即停止；拒绝协议参数与多节点 localhost |

新增测试：启动检查放行 torch218 的版本（及 lm-eval 0.4.9、datasets 5.x）、拒绝 datasets 3.6.0/4.6.1 与 lm-eval 0.4.8；
`pyproject` 中所有依赖都是 torch218 满足的下限，没有 `==`、上限，也没有 transformers/accelerate/openpyxl/sentencepiece；
离线缓存的 `Feature type ... not found` 与 `dataset_info.json` 解析失败给出指向环境而非缓存的报错；比较工具在任一版本不同时拒绝；
包内没有 `litgpt/utils/data/model` 顶层 import 和 `sys.path` 操作。

### 未验证

- 服务器上的 torch218 本身（真实 `PYTHONPATH`、editable litgpt 共存、conda 激活）。
- `majob.sh` 的 CUDA 自检与 CUDA 12.8 库路径在训练作业中的实际效果；GPU、NCCL/Gloo 多节点作业。
- 真实 Qwen3-1.7B-Base 权重、服务器上的离线 HF/RULER 缓存；`majob.sh --limit 4` 小样本、全量评测、与 SemanticLogKV 的真实样本对照均未执行。

## 2026-10-15：统一规则对齐（任务套件、majob.sh、离线与可追溯性）

环境：Linux CPU、Python 3.13.16、torch 2.14.1+cu130（仅 CPU 执行）、lm-eval 0.4.9、datasets 3.6.0、
transformers 4.57.6、tokenizers 0.22.2、pyarrow 26.0.0。

| 检查 | 结果 |
|---|---|
| `python -m pytest -q` | **55 passed**（同一环境下改动前为 50 passed） |
| 任务名解析：`configs/eval_suites.yaml` 的 47 个 main 与 3 个 NIAH 名称交给 lm-eval 0.4.9 `TaskManager` | 50 个全部可解析；`mmlu`、`ceval-valid` 为任务组，由 lm-eval 展开。只检查任务注册，没有读取数据 |
| `python -m h2okv.evaluate --config ... --suite main/niah --print-config` | 不加载权重与数据即可输出解析结果；main 阶段 `metadata=null`，NIAH 为 `[32768]` |
| `bash -n majob.sh` 与模拟解释器测试 | 单节点/双节点的 rank、world size、主节点 DNS、两阶段端口；main 失败返回原退出码且不启动 NIAH；改协议参数与多节点 localhost 在启动前报错 |

新增测试覆盖：项目内任务清单（47 main、其中 33 LongBench/LongBench-E、3 NIAH 仅 32768）、`--suite` 与 `--tasks`
互斥、main 阶段不传 NIAH metadata、结果与解析配置按运行编号命名且不覆盖历史文件、结果记录解释器/拓扑/离线变量、
比较工具在 limit 或窗口等协议字段不同时拒绝比较（缺字段的 SemanticLogKV 原始结果列为未核对）。

端到端测试仍是随机小模型 + 本地合成任务，经真实 lm-eval 0.4.9 `simple_evaluate` 跑通生成与选择题请求；
它验证请求语义与结果格式，不代表真实任务数据可读。

### 未验证（2026-10-15 记录）

- 真实 Qwen3-1.7B-Base 权重、服务器上的离线 HF/RULER 缓存（含各任务数据实际读取）。
- CUDA/SDPA 上的数值、显存与速度；NCCL/Gloo 多节点作业；`majob.sh` 在训练平台的真实执行。
- 任何全量评测结果；与 SemanticLogKV 的真实样本对照（`compare_results`）尚未执行。
