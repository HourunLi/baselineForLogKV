# 验证记录

只记录本分支实际执行过的检查。配置检查、CPU 局部测试与真实模型/全量评测分开表述。

## 2026-10-15：统一规则对齐（任务套件、majob.sh、离线与可追溯性）

环境：Linux CPU、Python 3.13.16、torch 2.14.1+cu130（仅 CPU 执行）、lm-eval 0.4.9、datasets 3.6.0、
transformers 4.57.6、tokenizers 0.22.2、pyarrow 26.0.0。

| 检查 | 结果 |
|---|---|
| `python -m pytest -q` | **60 passed**（同一环境下改动前为 55 passed） |
| 任务名解析：`configs/eval_suites.yaml` 的 47 个 main 与 3 个 NIAH 名称交给 lm-eval 0.4.9 `TaskManager` | 50 个全部可解析；`mmlu`、`ceval-valid` 为任务组，由 lm-eval 展开。只检查任务注册，没有读取数据 |
| `python -m pyramidkv.evaluate --config ... --suite main/niah --print-config` | 不加载权重与数据即可输出解析结果；main 阶段 `metadata=null`，NIAH 为 `[32768]` |
| `bash -n majob.sh` 与模拟解释器测试 | 单节点/双节点的 rank、world size、主节点 DNS、两阶段端口；main 失败返回原退出码且不启动 NIAH；改协议参数与多节点 localhost 在启动前报错 |

新增测试覆盖：项目内任务清单（47 main、其中 33 LongBench/LongBench-E、3 NIAH 仅 32768）、`--suite` 与 `--tasks`
互斥、main 阶段不传 NIAH metadata、结果与解析配置按运行编号命名且不覆盖历史文件、结果记录解释器/拓扑/离线变量、
比较工具在 limit 或窗口等协议字段不同时拒绝比较（缺字段的 SemanticLogKV 原始结果列为未核对）。

端到端测试仍是随机小模型 + 本地合成任务，经真实 lm-eval 0.4.9 `simple_evaluate` 跑通生成与选择题请求；
它验证请求语义与结果格式，不代表真实任务数据可读。

## 未验证

- 真实 Qwen3-1.7B-Base 权重、服务器上的离线 HF/RULER 缓存（含各任务数据实际读取）。
- CUDA/SDPA 上的数值、显存与速度；NCCL/Gloo 多节点作业；`majob.sh` 在训练平台的真实执行。
- 任何全量评测结果；与 SemanticLogKV 的真实样本对照（`compare_results`）尚未执行。
