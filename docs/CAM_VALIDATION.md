# CaM 验证记录

## 2026-10-15：项目独立化与统一评测协议

改动：移除分支内的 DMS 方法代码；移除运行时对相邻 `../litgpt` 的全部依赖（评测代码、实验 YAML、`eval.sh`、
权重转换），改用项目内私有评测快照、实验 YAML、任务清单与转换器；任务改为 47 个 main 入口与 3 个 NIAH
（仅 32768）分阶段运行；新增 `majob.sh`；评测必须显式选择 `--suite`/`--tasks`；结果记录 pyarrow、解释器、
CUDA/驱动与拓扑，解析配置按启动编号保存。

环境：Linux CPU，Python 3.13.16、torch 2.14.1+cu130（仅 CPU 执行）、transformers 4.57.3、lm-eval 0.4.9、
datasets 3.6.0、pyarrow 26.0.0；没有 CUDA，也没有相邻 `litgpt` 目录。

```bash
python -m pytest -q
```

结果：**52 passed**。改动前的同类测试直接导入 `../litgpt`，在没有该目录的环境中无法运行，因此没有同环境的
改动前对照数。在原有 CaM 算法/缓存/模型测试之外，新增或改写：

- 私有转换器：tiny LitGPT 模型 → HF，由参考快照的 HF→LitGPT 转换逐张量零误差还原，logits 一致。
- 准备好的 tiny Base 经真实私有加载的快照 `eval.py` 接入 CaM，执行继承来的选择题与生成请求；快照文件字节不变。
- 任务清单 47/33/13/3 与 NIAH `[32768]`；main 不传 NIAH metadata；三种后端任务输入一致。
- 配置边界：实验 YAML 必须在项目内；`--print-config` 只读项目内文件且不导入 torch/transformers/lm-eval；
  不选任务时报错；AST 检查无外部 `litgpt` 导入或 `sys.path` 操作；项目内不再包含 DMS。
- `majob.sh`：单/双节点 rank、world size、主节点 DNS、两阶段端口，main 失败返回原退出码且不启动 NIAH，
  改协议参数与多节点 localhost 报错；`scripts/evaluate_cam.sh` 的单机/多机参数。
- 输出隔离、结果写入 `resolved_eval` 与方法元数据、新版 datasets 缓存条目报错。

未验证：真实 1.7B CUDA/BF16 推理、32K 峰值显存/吞吐、服务器离线缓存与任务数据读取、多节点通信及任何任务分数。

## 2026-10-12：初始实现（历史记录，发生在独立化之前）

日期：2026-10-12；分支：`CaM`。macOS CPU，临时隔离环境 Python 3.11、PyTorch 2.9.1、Transformers 4.57.3、datasets 3.6.0、lm-eval 0.4.9。没有 CUDA、真实 1.7B 权重或服务器评测数据。

执行：

```bash
python -m pytest -q \
  tests/test_cam_cache.py tests/test_cam_model.py \
  tests/test_cam_evaluation.py tests/test_cam_config.py \
  tests/test_prepare_base.py tests/test_comparison_contract.py \
  tests/test_readonly_adapter.py tests/test_qwen3_base.py \
  tests/test_semanticlogkv_eval.py tests/upstream/test_dms_utils.py \
  tests/test_qwen3_cuda.py
```

**68 passed, 1 skipped，13.27 秒。** 跳过的是已有 DMS CUDA 检查；CaM 检查均在 CPU 执行。`git diff --check`、新 shell 的 `bash -n` 通过。

检查内容：

- CaM 逐 token 累计 attention、概率公式与边界、seeded Bernoulli、合并后真实删除 KV、scores 同步删除、zero sink 与多轮淘汰。
- Qwen3 Q/K normalization 与 raw K、GQA query-head 概率聚合、无淘汰时与原模型 logits 对齐（包括 head_dim 不等于 hidden_size/head_count 的配置）。
- 首次超预算输出仍包含 victim；未来 token 不改变过去输出；不同 chunk、请求重置保持可复现；独立 RNG 不改变全局随机状态。
- 模型 forward 严格检查连续位置和上下文边界，仅计算所需 logits；Full KV 对照、KV/FP32 分数实际张量字节统计、请求结束释放缓存。
- 从 tiny LitGPT Base 转换，检验 manifest 并实际加载 CaM，执行继承自参考 eval.py 的真实选择题与生成方法；另一次参考导入保持原生接口。
- 配置完整任务列表、NIAH 六种长度与当前 `litgpt/eval.sh` 逐项一致；三方输出合同、来源验证、错误标签移除和结果元数据。
- 子进程验证缓存环境在 Transformers/Hugging Face 导入前生效；对新入口实际执行 `--print-config`，库实际 cache 路径与环境设定一致，未加载权重。
- Shell 参数模拟覆盖单机、标准多机、ModelArts 地址解析，以及拒绝多机静默 standalone；这不是实际分布式 GPU 执行。
- 原有 DMS 离线测试及共享评测回归通过。未改动参考 `litgpt` 的源文件；原有 README/tutorials 未提交修改保留。

尚未验证：实际 Qwen3-1.7B CUDA/BF16 推理、32K 峰值显存/吞吐、服务器数据与任务精度、多卡通信。没有新增训练或产生正式评测结果。下一步仅做文档中的 Full KV / CaM 相同任务各 4 条样本检查，再决定是否扩大。
