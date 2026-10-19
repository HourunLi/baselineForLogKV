# Qwen3-1.7B-Base DMS baseline

在 `DMS` 分支接入 NVIDIA 官方 **Dynamic Memory Sparsification**，底模为 SemanticLogKV 当前实验所用的原始 Qwen3-1.7B-Base。直接从本项目实验配置 `configs/experiment_qwen1.7b_32k.yaml` 指定的 `ckpt_dir/lit_model.pth` 转换，不重新下载同名模型。冻结底模，仅训练每层、每个 KV head 的淘汰适配器；同一底模关闭淘汰后作为蒸馏 teacher。

这是一份可训练的实现，仓库不包含已经训练好的 1.7B DMS 权重。采用的是官方当前的 **adapter-only** 配方，不能表述为原论文全参数训练结果的复现，也不是 KVPress 的 `DMSPress` 包装器。

**项目边界：本项目独立运行。** 统一规则见 [AGENTS.md](AGENTS.md) 与 [Baseline 实现规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md)。SemanticLogKV 的评测协议（`eval.py` 及其 LitGPT 依赖）以私有快照放在 `models/_vendor/semanticlogkv/`（来源 `HourunLi/LogKV@4fd97c7`，Apache-2.0，来源与哈希见其 `SOURCE.json`）；实验配置、任务清单、Base 转换器均在本项目内。运行时不读取、不导入相邻 `litgpt`/SemanticLogKV 的源码、实验 YAML 或脚本，也不需要其 editable 安装。共享的只读资产只有原始 Base 权重/tokenizer 与离线评测缓存。

## 实现与来源

- 官方门控、延迟淘汰、FlexAttention 训练、压缩 prefill 和 paged KV decode 源码保留在 `dms/`；修正了 prefill 阶段漏计近期窗口的 CR 统计。
- Qwen3 模型、训练、导出和推理入口位于 `models/qwen3/`。结构从 Base 配置读取，无硬编码 8B 层数。
- 1.7B Base：28 层、hidden 2048、16 个 Q heads、8 个 KV heads、head_dim 128、绑定 embedding。新增可训练参数为 **516,096**。
- 官方源码固定于 `f27b47626d0d5886244d0fe1075e52d7d564ddc9`；来源与修改记录见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。许可证见 [LICENSE](LICENSE)。

## 环境：训练与评测分开

DMS 需要训练。训练沿用上游配方固定的依赖，评测在 SemanticLogKV 的评测环境 torch218 中进行，两者不混装（规范 4.3）。
当前环境的版本、路径与已知坑见 [Baseline 环境与避坑清单](docs/BASELINE_ENVIRONMENT.md)。

| | 训练环境（另建，不是 torch218） | 评测环境（SemanticLogKV 的 torch218） |
|---|---|---|
| 做什么 | `prepare_base`、`scripts/train.sh`、`extract` | `models.qwen3.evaluate`、`scripts/evaluate.sh`、`majob.sh`、`compare_results` |
| 依赖 | `python -m pip install -e '.[dms-train,test]'`：上游固定的 `transformers==4.57.3`、`accelerate==1.4.0`、`datasets==4.4.2` | **不安装本项目，也不往其中安装任何包**；从项目根目录用 `python -m` 运行 |
| 预检 | `python -m models.qwen3.check_env --training-only`（要求 transformers 4.57.3） | `python -m models.qwen3.check_env`（CUDA、BF16、FlexAttention `AuxRequest`、`flash_attn_with_kvcache`、transformers>=4.51） |

训练环境需要支持 BF16 的 NVIDIA GPU 与 PyTorch FlexAttention/Triton，上游推荐 `nvcr.io/nvidia/pytorch:25.11-py3`（Python 3.10–3.13）。
（未验证的选项）也可以 `conda create -n dms-train --clone torch218` 后在克隆出的环境里安装 `[dms-train]`，这样训练与评测共用同一份
torch/CUDA/FlashAttention；无论哪种方式都不要在 torch218 本身安装。当前实现不支持昇腾 NPU、MPS 或 CPU 的正式训练/推理。

评测侧的代码适配 torch218（2026-10-16：torch 2.11.0+cu128、transformers 4.51.3、lm-eval 0.4.11、datasets 4.8.4、flash-attn 2.8.3，
没有 accelerate、sentencepiece）：

- 评测入口在 import 评测框架之前只检查 `lm-eval>=0.4.9`、`datasets>=4.7`（共享缓存中 IFEval 条目的 `Json` 特征类型从 4.7 起可读），
  不满足时报错并提示回到 torch218；精确版本写进每个结果的 `comparison_contract.versions`，`compare_results` 要求双方一致。
- DMS 代码按 transformers 4.57.3 写成，用到了 4.51 没有的 `CacheLayerMixin`/分层 `Cache`、`masking_utils`、`GradientCheckpointingLayer`
  与 `TransformersKwargs`。[`dms/transformers_compat.py`](dms/transformers_compat.py) 在已安装的 transformers 提供这些名字时直接用它的，
  否则用从 4.57.3 原样复制的类或等价的替代；训练环境因此运行上游原类，不受影响。模型类另外显式给出 `config_class` 与
  embedding 访问器（4.51 没有默认实现，否则 `lm_head` 不会与 embedding 绑定）。
- 训练环境导出的 student 在两个环境中的评测行为用 [`tests/cross_env_check.py`](tests/cross_env_check.py) 核对：
  分块 prefill 与逐 token decode 的 logits、生成 token 和每一步的 KV 条目数都逐元素相同（记录见 [验证记录](docs/VALIDATION.md)）。
- `pyproject.toml` 的核心依赖与 `[eval]` 只有 torch218 满足的下限；上游的版本固定只在 `[dms-train]` 中。
- 若曾在 torch218 中执行过 `pip install -e .` 或安装过旧的 `[train]`/`[eval]` extra，可用
  `python -m pip uninstall semanticlogkv-dms-baseline` 撤销本项目的 editable 条目；旧 extra 若改动过 transformers、datasets、lm-eval 等，
  按环境清单第 1 节导出快照与版本表核对，有差异时交由环境维护者恢复，不在共享环境里自行重装。

**lm-eval 版本决定任务定义。** torch218 的 lm-eval 0.4.11 与 0.4.9 的 LongBench 数据源（`THUDM/LongBench` → `Xnhyacinth/LongBench`）、
prompt 字段（`{{input}}` → `{{question}}`）和解码方式（`do_sample: True` → `False`，即贪心）不同，arc、ceval、ifeval 与 RULER 不变。
只比较同一 lm-eval 版本下的结果；SemanticLogKV 来自其他版本的旧结果须在 torch218 中重跑。

## 准备同一份 Base

在训练环境中执行（输出是训练的输入）：

```bash
python -m models.qwen3.prepare_base
```

默认读取项目内 `configs/experiment_qwen1.7b_32k.yaml`（SemanticLogKV 实验 `arc_attach_alpha_beta_k12_b128_2k` 继承链的平铺快照，只保留 Base、32K 窗口和 LogKV 对照参数），使用其中 **原始 `ckpt_dir`**（可用 `DMS_BASE_CHECKPOINT` 覆盖），不使用 SemanticLogKV CPT 的输出。`--experiment-config` 只接受项目内文件。输出到 `outputs/semanticlogkv_base`，两份训练配置均指向此处。输出目录已存在时拒绝覆盖。

转换使用项目内 `models/qwen3/convert_lit_to_hf.py`（改编自 LitGPT 0.5.12 的 Qwen3 映射），严格校验参数、绑定 embedding 和模型配置，原样复制 tokenizer，保存来源、实验 YAML、转换器及输出文件 SHA-256。训练启动前验证已转换文件；来源信息随 DMS 模型导出，并在正式评测前与当前比较 Base 再核对。没有 manifest 的旧模型不能直接进入正式对比；早先用相邻 LitGPT 转换的产物（manifest v1）仍按同样哈希校验。

当前配置的权重与测试缓存是服务器 `/home/ma-user/...` 路径，本地 Mac 未存放这些真实文件。应在同样挂载这些目录的 CUDA 机器执行；不会因文件缺失自动下载替代版本。完整说明见 [评测对齐记录](docs/SEMANTICLOGKV_ALIGNMENT.md)。

## 先做小规模检查

不下载权重的 CPU 检查，两个环境都在项目根目录运行，不设 `PYTHONPATH`：

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

验证配置、纯文本格式、1.7B 参数尺寸、底模加载与冻结、teacher 数学一致性、适配器梯度、保存与重载、评测加载器、启动器与依赖声明。
CPU attention 检查使用 eager 执行，不能证明 CUDA 内核可用。在 torch218 中，依赖训练引擎的 2 项测试标明“training environment”跳过；
在训练环境中，需要 lm-eval 的 1 项跳过。

两个环境之间的行为用跨环境检查核对（CPU 即可；DIR 为任意新目录）：

```bash
<训练环境 python> -m tests.cross_env_check export DIR/student
<训练环境 python> -m tests.cross_env_check run DIR/student DIR/train.json
<torch218 python> -m tests.cross_env_check run DIR/student DIR/eval.json
<torch218 python> -m tests.cross_env_check compare DIR/train.json DIR/eval.json   # 应输出 IDENTICAL
```

CUDA 内核检查同样不下载权重，使用随机小模型，序列 1024，超过 512 的保护窗口：

```bash
python -m pytest -q tests/test_qwen3_cuda.py
```

准备上述 Base 后，使用真实 1.7B 权重跑两步，仅读取仓库内合成文本，不下载 OpenR1 数据集：

```bash
bash scripts/train.sh configs/qwen3_1.7b_base_smoke.yaml
```

这是流程检查，**不能用合成数据产出的 checkpoint 评价方法质量**。配置默认单 GPU、1024 tokens、2 步、不启用 FSDP；真实峰值显存需在目标 GPU 上验证。输出为 `outputs/qwen3_1.7b_base_smoke/student_model/`。

## 训练

在训练环境中执行；`scripts/train.sh` 先做 `check_env --training-only`，不是训练环境时直接报错。

```bash
# 单机，GPU 数量按机器设置；CUDA_VISIBLE_DEVICES 可选择具体设备
NPROC_PER_NODE=1 bash scripts/train.sh configs/qwen3_1.7b_base.yaml
```

脚本先由单进程准备数据，再使用 torchrun 启动训练。主配置基于官方 1.7B 调试配方：8K 上下文、544 个优化器步、每卡 batch 1、梯度累积 16、学习率 3e-5、目标 CR 从 1 到 16（第 510 步达到），保留 512-token 延迟窗口。采用 FSDP、CPU offload 和逐层 activation checkpointing；仍需足够的 GPU 与主存，未给出未经实测的显存承诺。

GPU 数量变化会改变总 token 预算：近似为 `steps × GPU数 × 每卡batch × 梯度累积 × 序列长度`，不能只对齐步数。这里的 `train_samples: 4000` 是数据混合索引长度，训练可能重复采样，并不代表总共只见过 4000 个样本。

`model.name` 必须指向 `prepare_base` 产生并带完整 manifest 的本地目录。更换原始 Base 时重新准备独立输出，不能直接指向另一份 HF 权重。训练上下文目前仍为官方 8K 配方，正式评测统一为 SemanticLogKV 当前的 32K 窗口；本次没有改变训练语料和 token 预算。

### 数据格式

默认继承 OpenR1-Math-220k 的正确解答过滤，**将问题与解答按纯文本拼接并追加 EOS**，不添加 chat role token 或 thinking 模板。原始解答自带的文字标记不会被额外清洗。该语料与 SemanticLogKV CPT 语料尚未对齐。

接入相同 CPT 文本时，删除 `data.blend`，替换为：

```yaml
data:
  train_file: /absolute/path/train.jsonl
  train_samples: 4000
  max_length: 8192
  concat_always_start_new: false
  process_vocab_using_chunk: 4096
  tokenizer_kwargs:
    chat_template: "{{ messages | map(attribute='content') | join('\\n\\n') }}{{ eos_token }}"
```

每行格式为 `{"text": "一段非空原始文本"}`。加载、分词、拼接仍复用官方数据管线；仅保留完整上下文，最后不足一段的尾部会丢弃。`concat_always_start_new: true` 沿用上游行为，会额外丢弃每次拼满上下文后的文档溢出部分；使用 `false` 可连续利用文本。本地数据不使用官方按路径的中间缓存，避免同路径内容修改后读到旧结果；Hugging Face 自身仍按文件指纹缓存。

### 恢复与导出

YAML `hf_trainer.resume_from_checkpoint` 设置为 `auto` 时选择最新 checkpoint；没有 checkpoint 则开始新训练。也可填写明确目录，目录不存在会报错。恢复时应使用同一训练配置和数据。

训练结束自动收集完整 student 权重，由主进程写入 `student_model/`。保存的是完整模型与适配器、tokenizer、模型配置和实现文件；不是只有小型 adapter 文件。

中间训练 checkpoint 可单进程导出：

```bash
python -m models.qwen3.extract \
  --config outputs/qwen3_1.7b_base/config.yaml \
  --checkpoint outputs/qwen3_1.7b_base/checkpoint-34 \
  --output outputs/qwen3_1.7b_base/student-step34
```

该入口读取完整 `pytorch_model.bin`；本仓库 FSDP 配置要求 FULL_STATE_DICT。它需要原始底模和 checkpoint，不能直接读取任意外部 sharded checkpoint。

## 推理与评测

```bash
python -m models.qwen3.infer \
  --model outputs/qwen3_1.7b_base/student_model \
  --prompt 'The key observations are' --max-new-tokens 128
```

此 `infer` 命令仅用于独立 HF 续写诊断，不作为横向评测入口。输入不应用聊天模板；每次独立生成建立新 DMS cache。`--disable-eviction` 可检查同权重关闭淘汰的行为。输出逻辑 KV 压缩比不等于实际显存压缩比。

`final_cr` 是**训练目标**。模型配置的 `dms_cr` 仅记录目标，修改它不会在推理时强制达到某个压缩倍数。实际压缩比随输入和保护窗口变化，应实测并与质量一同报告。

正式横向评测在 torch218 中执行，两种方法使用同一入口。每次必须选 `--suite main|niah|longbench`（项目内 `configs/eval_suites.yaml`：47 个 main 入口、其中 33 个 LongBench/LongBench-E，3 个 NIAH 仅 32768；输出进入 `<output>/<suite>/`）或诊断用 `--tasks`：

```bash
source /home/ma-user/anaconda3/bin/activate torch218
cd /data/lihourun/baselineForLogKV/DMS          # 本分支的检出目录；不安装任何包
python -m models.qwen3.check_env                 # GPU 机器上：CUDA、BF16、FlexAttention、FlashAttention 接口
python -m models.qwen3.evaluate --model outputs/qwen3_1.7b_base/student_model --suite main --print-config
bash scripts/evaluate.sh outputs/qwen3_1.7b_base/student_model --suite niah --limit 4
# 使用同一配置、任务、样本数检查 SemanticLogKV CPT checkpoint
bash scripts/evaluate.sh /path/to/semanticlogkv/checkpoint --backend logkv --suite niah --limit 4
```

模型窗口 32K、batch 1。main 阶段不传 NIAH metadata，与参考脚本一致。删除 `--limit` 才是任务全量。`NPROC_PER_NODE` 控制每卡一份模型，默认 1。启动器私有加载项目内快照的 `eval.py`，在该私有评测实例中接入 DMS 子类，复用 tokenizer、生成循环、评分与指标统计。评测强制离线；map/filter 预处理写进程私有临时文件。共享离线缓存只读：报 `Feature type 'List'/'Json' not found` 说明安装的 datasets 比写缓存的版本旧（环境被降级，例如误在训练环境评测），应回到 torch218；某个条目的 `dataset_info.json` 不是合法 JSON 时报错并停止，不移动、不改写缓存条目，见环境清单第 4 节。

### 训练平台作业：majob.sh

`majob.sh` 是完整评测入口，**每个节点执行一次**：先 main（47 个入口）再 NIAH（3 个入口，仅 32768），两阶段各自一次 `torch.distributed.run`，main 失败即返回非零且不启动 NIAH。它只评测已导出的模型，必须给 `--model`，从不训练；拒绝 `--suite/--tasks/--evaluation-config/--experiment-config/--print-config`。

```bash
bash majob.sh configs/experiment_qwen1.7b_32k.yaml --model outputs/qwen3_1.7b_base/student_model \
  --limit 4 --output outputs/dms-32k-smoke
bash majob.sh configs/experiment_qwen1.7b_32k.yaml --model outputs/qwen3_1.7b_base/student_model \
  --output outputs/dms-32k-full
```

平台变量与 chunkKV 等分支一致：`MA_NUM_GPUS`（每节点进程数，缺省 `NPROC_PER_NODE`/`LOCAL_WORLD_SIZE`/本节点可见的 CUDA 设备数，超过可见设备数时在启动 torchrun 前报错）、`MA_NUM_HOSTS`、`VC_TASK_INDEX`、`MA_VJ_NAME`/`MA_TASK_NAME`/`MA_MASTER_INDEX`（主节点 DNS，多节点禁止 localhost）、`MASTER_PORT`（只用于多节点：main 用 +1、NIAH 用 +2，可由 `EVAL_MASTER_PORT`/`NIAH_MASTER_PORT` 覆盖；单节点每个阶段用 `--standalone` 绑定空闲端口）。激活环境后、任何 CUDA 调用之前，脚本照搬 SemanticLogKV `eval.sh`/`majob.sh` 的 CUDA 12.8 `PATH`/`LD_LIBRARY_PATH` 与 NCCL 设置，并用同一解释器执行一次 `torch.cuda.init()`，失败时打印加载的 libcuda 与 `LD_LIBRARY_PATH` 后退出；`scripts/evaluate.sh` 使用同样的库路径。平台上建议设置 `TMPDIR=/tmp`。`DMS_PYTHON` 直接指定解释器，否则激活 `/home/ma-user/anaconda3/bin/activate ${DMS_CONDA_ENV:-torch218}`；环境名不代表实际版本，以结果记录为准。脚本导出 `LITGPT_EXPECTED_WORLD_SIZE`，快照评测器据此检查 rendezvous。

结果包含 `comparison_contract`、`resolved_eval` 和 harness 逐样本日志，可自动核对：

```bash
python -m models.qwen3.compare_results /path/to/dms/niah/eval_results_TIMESTAMP.json \
  /path/to/semantic/niah/eval_results_TIMESTAMP.json
```

只有 Base 来源、tokenizer、评测代码、运行时版本（含 pyarrow）、任务设置及实际样本/prompt/目标/生成参数一致才通过；后端、实际评测权重与预算作为方法差异单独列出。每次启动写 `resolved_eval_<launch_id>.yaml`，不覆盖历史记录。旧评测结果若缺这些记录，需要先补小样本对照，不能直接宣称等价。

## 论文实验口径

本实现名称建议记录为 **DMS-adapter / Qwen3-1.7B-Base**。记录底模版本、语料、训练 token 数、适配器参数量、目标与实测 CR、上下文长度。与 SemanticLogKV 比较时，需要另外明确语料和训练预算是否一致；当前默认 OpenR1/8K 配方本身不构成公平性证明。

当前交付包含实现、项目内评测协议和离线 CPU 验证，不包含正式训练或评测结果。检查范围与环境限制见 [验证记录](docs/VALIDATION.md)。CPU 检查不覆盖 CUDA 编译、FlashAttention decode、FSDP 多卡执行、真实离线缓存、多节点作业或真实 1.7B 精度。
