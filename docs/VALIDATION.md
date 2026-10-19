# 本地验证记录

## 2026-10-19（续）：服务器 torch218 首次运行与修复

服务器 torch218（GPU 机器，`/data/lihourun/baselineForLogKV/DMS`，`2a3c18c`）上
`env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q`：**98 passed、2 skipped、1 failed**（用户运行）。

- 通过的包括上游 10 项 CUDA 测试（分块 prefill 与 prefill 后生成，对照参考实现）：编译的 FlexAttention prefill 与
  `flash_attn_with_kvcache` 分页 decode 在 torch218 中可运行。这是 GPU 推理路径的第一次实测。
- 跳过的 2 项是依赖训练引擎的测试（training environment）。
- 失败的是 `tests/test_qwen3_cuda.py`（一步蒸馏后用 HF `generate()` 做 prefill/decode）：训练前向传入全 1 的 `attention_mask`，
  兼容层的 `create_causal_mask` 替代把任何非 `None` 的 mask 都当作 padding 拒绝。

修复：替代实现对没有 padding 的输入（没有 mask 或全 1 二维 mask）返回 `None`，与 4.57.3 在 SDPA 下的返回值相同，只拒绝真正的 padding；
复制的 `Cache` 在旧版 transformers 中继承已安装的 `Cache`，模型类加 `_supports_cache_class`，否则 4.51 的 `generate()` 不接受 `DMSCache`
（这两处在修完 mask 后依次暴露）。正式评测走 vendor 评测器自己的生成循环，不经过 HF `generate()`；受影响的是这个测试和 `infer.py`。

验证（开发容器 CPU 副本，环境同上一节）：

| 检查 | 结果 |
|---|---|
| `test_qwen3_cuda.py` 的 CPU 镜像（同样的蒸馏一步与两次 `generate()`；CUDA kernel 换成上游测试的参考实现、未编译的 FlexAttention，训练噪声的 `fork_rng` 不涉及 CUDA） | 修复前在 torch218 副本中依次失败于 mask、`isinstance` 与 `_supports_cache_class`；修复后两个副本都通过 |
| 同上，两个副本从同一组初始权重出发 | loss、生成 token 与压缩比**完全相同**。若各自用 `Qwen3ForCausalLMDMS(config)` 的默认初始化则不同：4.57.3 的 `PreTrainedModel` 有默认 `_init_weights`，4.51.3 没有。正式流程不依赖随机初始化（训练在训练环境从准备好的 Base 加载，评测加载完整 checkpoint） |
| torch218 副本 `python -m pytest -q` | **90 passed、13 skipped**（训练引擎 2 项、CUDA 11 项） |
| 训练环境副本 | **90 passed、13 skipped**（lm-eval 1 项、CUDA 11 项、已有 masking_utils 时跳过的旧版 mask 测试 1 项） |
| `tests/cross_env_check.py` | 仍逐元素相同 |

修复后尚未在服务器上重跑；`test_qwen3_cuda.py` 在真实 CUDA kernel 上的结果以服务器重跑为准。

## 2026-10-19：评测适配 torch218（transformers 4.51.3、lm-eval 0.4.11、datasets 4.8.4），训练环境分开

改动：评测在 SemanticLogKV 的 torch218 中运行，不安装任何包；训练沿用上游固定的 transformers 4.57.3、accelerate 1.4.0、
datasets 4.4.2，只装在另建环境（`[dms-train]`）。评测入口只检查 `lm-eval>=0.4.9`、`datasets>=4.7`；`dms/transformers_compat.py`
补齐 4.51 缺少的 `CacheLayerMixin`/分层 `Cache`（从 4.57.3 原样复制）、`create_causal_mask`、`GradientCheckpointingLayer`、
`TransformersKwargs`；模型类显式写 `config_class` 与 embedding 访问器；`check_env` 分训练/评测要求；`pyproject` 只留 torch218 满足的下限；
`majob.sh` 与 `scripts/evaluate.sh` 按 chunkKV 的做法对齐（可见卡数、CUDA 12.8 库路径与自检、单节点 `--standalone`）。DMS 的门控、
淘汰、缓存与注意力算法，以及 vendor 的评测快照均未改动（快照与 chunkKV 分支逐文件相同，仅命名空间、头注释与 `DMS_HF_CACHE` 不同）。

环境（开发容器，CPU，均为 Python 3.10.20、torch 2.11.0 的 PyPI 默认 CUDA 版，只在 CPU 上执行；没有 FlashAttention）：

- **torch218 副本**：transformers 4.51.3、tokenizers 0.21.4、lm-eval 0.4.11、datasets 4.8.4、huggingface-hub 0.36.2、pyarrow 23.0.1、
  numpy 2.2.6、pandas 2.3.3、xlsxwriter 3.2.9、lightning 2.6.1、jsonargparse 4.41.0、litdata 0.2.61、tensorboard 2.20.0；
  与 torch218 一样没有 accelerate、sentencepiece、openpyxl。按环境清单第 1 节的版本表逐个安装，没有完整 freeze。
- **训练环境副本**：transformers 4.57.3、accelerate 1.4.0、datasets 4.4.2、tokenizers 0.22.1、lightning 2.6.1、jsonargparse 4.41.0；
  没有 lm-eval。

| 检查 | 结果 |
|---|---|
| 改动前的 `origin/DMS`（6891e07），torch218 副本中 `python -m pytest -q` | **9 个测试文件收集失败**（`cannot import name 'CacheLayerMixin' from 'transformers'`），与服务器 torch218 上看到的报错相同 |
| 改动前，训练环境副本 | 76 passed、12 skipped（缺 lm-eval 1 项、需要 CUDA 11 项） |
| 改动后，torch218 副本 `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q` | **88 passed、13 skipped**：依赖训练引擎的 2 项标明 training environment 跳过，需要 CUDA 的 11 项跳过 |
| 改动后，训练环境副本 | **89 passed、12 skipped**（缺 lm-eval 1 项、需要 CUDA 11 项） |
| `tests/cross_env_check.py`：训练环境副本导出一个 adapter 非零的小型随机 student（项目的 `extract_student_model`，transformers 4.57.3 写出），两个副本分别按评测器的加载步骤与 `DMSModel.forward`/`kv_compression_snapshot` 跑 96 token 的 3 块 prefill 与 12 步贪心 decode | **逐元素相同**：96×64 的 prefill logits、每步 decode logits 与 token、每步 KV 条目数（结束时 432 中保留 285，确有淘汰）。CPU 上用上游测试的参考 FlashAttention 替代与未编译的 FlexAttention |
| 反向对照：去掉新增的 embedding 访问器后在 torch218 副本重跑 | 报 `lm_head is not tied`；正式加载器同样会因缺 `lm_head.weight` 拒绝（`test_shared_loader_ties_lm_head_with_the_installed_transformers`） |
| `--print-config` | 系统 Python 3.13（只有 PyYAML，没有 torch）中正常输出；原有测试在禁止 import torch/transformers/lm_eval/datasets 的情况下通过 |

新增或改写的测试：`majob.sh` 的可见卡数默认值、超出可见卡数与 CUDA 初始化失败时不启动 torchrun、`LD_LIBRARY_PATH` 顺序、单节点
`--standalone`、多节点 rank/端口/主节点 DNS；启动检查放行 torch218 的版本、拒绝训练环境的 datasets 4.4.2 与过旧版本；`pyproject` 的核心依赖、
`[test]`、`[eval]` 都是 torch218 满足的下限，上游版本固定只在 `[dms-train]`；`dataset_info.json` 解析失败与新版 datasets 缓存条目的报错；
评测加载器在当前 transformers 下加载完整、`lm_head` 绑定、bfloat16。

### 未验证

- 服务器上的 torch218 本身；flash-attn 2.8.3 在 torch218 中是否可 import、`flash_attn_with_kvcache` 与 FlexAttention 在 A800 上的执行。
- 任何 GPU 路径：编译的 FlexAttention prefill、FlashAttention paged decode、bfloat16 CUDA 数值；`tests/test_qwen3_cuda.py` 与上游 CUDA 测试。
- 训练环境本身尚未在服务器上建立；真实 1.7B Base 的 `prepare_base`、训练、导出均未运行，因此还没有可评测的 DMS student。
- `majob.sh` 在训练平台的执行、离线缓存的真实读取、`--limit 4` 小样本、全量评测与 SemanticLogKV 的对照。

## 2026-10-15：项目独立化与统一评测协议

改动：移除运行时对相邻 `../litgpt` 的全部依赖（评测代码、实验 YAML、权重转换），改用项目内私有快照、
实验 YAML、任务清单与转换器；新增 `majob.sh`（main 与 32K NIAH 分阶段，失败停止）；评测必须显式选择
`--suite` 或 `--tasks`；结果记录 pyarrow、解释器、CUDA/驱动与拓扑，解析配置按启动编号保存。

环境：Linux CPU，Python 3.13.16、torch 2.14.1+cu130（仅 CPU 执行）、transformers 4.57.3、accelerate 1.4.0、
lm-eval 0.4.9、datasets 3.6.0、pyarrow 26.0.0、lightning（随私有快照依赖安装）。没有 CUDA、FlashAttention，
也没有相邻 `litgpt` 目录。

```bash
python -m pytest -q -m "not cuda"
```

结果：**77 passed, 10 skipped, 1 deselected**。跳过的 10 项为上游 `test_prefill_and_generate.py` 中需要 CUDA
的参数化用例，deselect 的是 `cuda` 标记测试。改动前的评测/转换测试直接导入 `../litgpt`，在没有该目录的
环境中无法运行，因此没有同环境的改动前对照数。新增或改写的检查：

- 私有转换器：tiny LitGPT 模型 → HF，逐张量由参考快照的 HF→LitGPT 转换零误差还原；HF/LitGPT logits 一致；
  manifest 记录转换器哈希与项目实验 YAML。
- 配置边界：继承离开顶层 YAML 目录报错；`--experiment-config` 指向项目外文件时报错。
- 真实私有加载快照 `eval.py`（lm-eval 0.4.9、datasets 3.6.0）并接入 DMS 子类，快照文件字节不变、原生类与
  统计函数身份不变。
- 任务清单 47/33/3、main 不传 NIAH metadata、NIAH 仅 32768；`--print-config` 只读项目内 YAML/源码且不导入
  torch/transformers/lm-eval；不选任务时报错。
- AST 检查项目 `models/` 下没有 `litgpt`/`utils`/`data` 顶层导入或 `sys.path` 操作，脚本与配置不含 `../litgpt`。
- `majob.sh`：单/双节点 rank、world size、主节点 DNS、两阶段端口，main 失败返回原退出码且不启动 NIAH，
  缺 `--model` 或改协议参数时报错。
- 输出隔离、结果写入 `resolved_eval`、新版 datasets 缓存条目报错且恢复转换缓存开关。

未验证：真实 1.7B Base 指纹与转换、CUDA/FlashAttention/FlexAttention、真实训练、服务器离线缓存与任务
数据读取、多节点作业及任何任务分数。

## 2026-10-12：初始实现（历史记录，发生在独立化之前）

分支：`DMS`。以下检查当时依赖相邻 `litgpt` 可编辑安装，不能作为当前代码的通过记录。

环境：macOS，Python 3.11.0、PyTorch 2.9.1、Transformers 4.57.3、accelerate 1.4.0。首次训练组件检查使用 datasets4.4.2；评测对齐检查切换到 datasets3.6.0、lm-eval0.4.9、LitGPT editable 核心包。依赖装在临时隔离环境，未修改系统 Python。没有可用 CUDA，也没有安装 FlashAttention。

## 已执行

```bash
python -m pytest -q tests/test_qwen3_base.py tests/upstream/test_dms_utils.py tests/test_qwen3_cuda.py
```

首次结果：**15 passed, 1 skipped**，4.14 秒。CUDA 检查因无 CUDA 设备跳过。

覆盖内容：

- Base 纯文本模板与两份配置；1.7B 的完整结构在 meta device 上构造，确认 516,096 个适配器参数以及 embedding 绑定。
- 从本地随机小型原始 Qwen3 权重加载 DMS，确认权重保留、门控投影置零和底模冻结。
- 小模型 teacher 输出与原始 Qwen3 一致；eager CPU attention 前向、反向产生有限且非零的适配器梯度，底模没有梯度。
- BF16 导出、AutoModel 动态加载、适配器值与绑定权重恢复；导出不修改训练模型 dtype。
- 分布式导出的接口模拟：每个 rank 参与收集，只有主进程写文件。**这不是实际 FSDP 多卡验证。**
- 本地 JSONL 分词拼接、空上下文报错、恢复路径选择、压缩目标日程。
- prefill CR 统计包括近期窗口；无淘汰时为 1，迁入 paged cache 后也保持正确统计。
- 上游门控函数的形状、二值前向、温度和梯度检查。

此外完成 editable 安装，在仓库外运行训练与推理入口的 `--help`，Python 与 shell 语法检查、本地文档链接检查。环境预检按预期拒绝无 CUDA 的正式训练，没有静默切换成其他算法。

## 本次评测对齐检查

```bash
python -m pytest -q tests/test_qwen3_base.py tests/test_prepare_base.py tests/test_semanticlogkv_eval.py tests/test_comparison_contract.py tests/test_readonly_adapter.py tests/upstream/test_dms_utils.py tests/test_qwen3_cuda.py
```

撤回参考仓库接点、改为 baseline 自身兼容接口后的结果：**39 passed, 1 skipped**，5.80秒；包括前述原有检查。新增验证覆盖：

- 用真实 tiny LitGPT 模型执行官方 Qwen3 映射，逐张量零误差比较，HF/LitGPT logits 一致；tied head 或结构错误拒绝。
- Base 来源、转换权重、模型配置、tokenizer 哈希验证；篡改拒绝，禁止覆盖输出；来源信息经过 DMS 导出/重载保留。
- 从转换后的 tiny Base 经官方 student 加载、保存，再用共享评测 DMS loader 严格重载；仅跳过 CUDA 环境检查，不执行 GPU 前向。
- 从当前 `litgpt/eval.py` 抽取真实请求方法，调用真实 LitGPT generate，仅用 fake HF 计算代替 GPU attention；检查绝对位置、logits 切片、截断、BOS、EOS、当前 until 行为、rolling 窗口和最后未回填 token 的计数。
- DMS prefill/decode 两区 KV 计数与共享比率；两后端配置相同；同名任务但 prompt/生成参数/环境不同，结果核对拒绝。
- 仓库边界：私有加载/接入 DMS 前后，参考文件字节一致，原生 `litgpt.kv_compression` 函数身份不变；另一次加载仍是原生 LogKV 类，请求方法保持直接继承。
- 输出隔离：原入口的临时 JSON 只写入 baseline 输出目录，禁止输出到参考仓库；缺少本次新结果时拒绝给旧结果补来源信息。

实际运行 DMS 和 LogKV 两种后端 `--print-config` 均成功；确认库实际使用的 HF cache 与共享环境设置相同，解析任务为 single2/3、窗口32768。完成依赖一致性检查（pip check）、Python 编译和 shell 语法检查。

此前对共享 LitGPT 运行压缩统计、请求处理、HF缓存别名、缓存诊断、通信测试：最初 **22 passed、2 skipped、2 failed**。其中 Gloo 失败来自沙盒禁止回环端口，允许本机通信后单独重跑 **1 passed**。剩余挂载诊断测试依赖 Linux `/proc/self/mountinfo`，在 macOS 报 `KeyError: mount`；已逐函数 AST 对比确认其实现与原 HEAD 完全一致。此处保留历史检查结果，不将整组选定测试表述为全通过。

边界修正完成后，`git diff --exit-code -- eval.py litgpt/kv_compression.py docs/evaluation-metrics.md` 在参考仓库返回0；这三处新增内容已撤回。其余原本存在的 README/tutorials 修改保留，没有做全仓库重置。

## 尚未验证

- 没有下载或加载完整 Qwen3-1.7B-Base 权重，没有得到训练完成的适配器。
- 未执行 CUDA/Triton/FlexAttention 编译、FlashAttention decode、真实单卡训练或 FSDP 多卡训练。
- 未测显存、吞吐、任务精度，未运行 RULER。未完整执行上游所有测试。
- 未验证真实服务器 Base 的指纹和缓存样本内容；代码会在实际运行时检查来源并保存逐样本记录。CPU facade 测试不等于真实 DMS CUDA 推理验证。

下一步在目标 CUDA 机器准备同一份 Base，检查 CUDA 内核并完成两步训练/导出链路；已有正式适配器时，优先通过统一入口运行双方各4样本，再用 `compare_results` 核对。合成试跑模型不用于报告精度。
