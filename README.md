# Qwen3-1.7B-Base DMS baseline

在 `DMS` 分支接入 NVIDIA 官方 **Dynamic Memory Sparsification**，底模为 SemanticLogKV 当前实验所用的原始 Qwen3-1.7B-Base。直接从本项目实验配置 `configs/experiment_qwen1.7b_32k.yaml` 指定的 `ckpt_dir/lit_model.pth` 转换，不重新下载同名模型。冻结底模，仅训练每层、每个 KV head 的淘汰适配器；同一底模关闭淘汰后作为蒸馏 teacher。

这是一份可训练的实现，仓库不包含已经训练好的 1.7B DMS 权重。采用的是官方当前的 **adapter-only** 配方，不能表述为原论文全参数训练结果的复现，也不是 KVPress 的 `DMSPress` 包装器。

**项目边界：本项目独立运行。** 统一规则见 [AGENTS.md](AGENTS.md) 与 [Baseline 实现规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md)。SemanticLogKV 的评测协议（`eval.py` 及其 LitGPT 依赖）以私有快照放在 `models/_vendor/semanticlogkv/`（来源 `HourunLi/LogKV@4fd97c7`，Apache-2.0，来源与哈希见其 `SOURCE.json`）；实验配置、任务清单、Base 转换器均在本项目内。运行时不读取、不导入相邻 `litgpt`/SemanticLogKV 的源码、实验 YAML 或脚本，也不需要其 editable 安装。共享的只读资产只有原始 Base 权重/tokenizer 与离线评测缓存。

## 实现与来源

- 官方门控、延迟淘汰、FlexAttention 训练、压缩 prefill 和 paged KV decode 源码保留在 `dms/`；修正了 prefill 阶段漏计近期窗口的 CR 统计。
- Qwen3 模型、训练、导出和推理入口位于 `models/qwen3/`。结构从 Base 配置读取，无硬编码 8B 层数。
- 1.7B Base：28 层、hidden 2048、16 个 Q heads、8 个 KV heads、head_dim 128、绑定 embedding。新增可训练参数为 **516,096**。
- 官方源码固定于 `f27b47626d0d5886244d0fe1075e52d7d564ddc9`；来源与修改记录见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。许可证见 [LICENSE](LICENSE)。

## 环境：训练与评测都在 torch218

训练与评测都在 SemanticLogKV 的环境 torch218 中运行，**不安装本项目，也不往其中安装任何包**，从项目根目录用 `python -m` 运行。
当前环境的版本、路径与已知坑见 [Baseline 环境与避坑清单](docs/BASELINE_ENVIRONMENT.md)（2026-10-16：torch 2.11.0+cu128、
transformers 4.51.3、lm-eval 0.4.11、datasets 4.8.4、flash-attn 2.8.3，没有 accelerate、sentencepiece）。需要支持 BF16 的 NVIDIA GPU
与 PyTorch FlexAttention/Triton；不支持昇腾 NPU、MPS 或 CPU 的正式训练/推理。

| | 训练 | 评测 |
|---|---|---|
| 入口 | `prepare_base`、`scripts/train.sh`、`extract`；`majob.sh --train` 训练后接着评测 | `models.qwen3.evaluate`、`scripts/evaluate.sh`、`majob.sh --model`、`compare_results` |
| 预检 | `python -m models.qwen3.check_env --training-only`（CUDA、BF16、FlexAttention、transformers>=4.51） | `python -m models.qwen3.check_env`（另需 `flash_attn_with_kvcache`） |

- **训练不用 HF Trainer/accelerate**（torch218 没有 accelerate）。[`dms/training/loop.py`](dms/training/loop.py) 是本项目写的训练循环，
  对上游配方用到的设置重现 transformers 4.57.3 Trainer + accelerate 1.4.0 的行为：数据顺序与多进程切分、梯度累积与 loss 缩放、
  梯度裁剪、fused AdamW、常数学习率、DMS 噪声种子与压缩计划、bf16 autocast、日志，以及 FSDP（bf16 参数/梯度规约/缓冲、CPU offload、
  每个 decoder 层一个单元、逐层非重入 activation checkpointing，参数与 accelerate 由同一 YAML 得出的一致）。YAML 的 `trainer:` 段沿用
  TrainingArguments 的字段名和含义，循环没有实现的取值直接报错，不会被忽略。损失、蒸馏前向、门控与数据管线仍是官方代码。
  它不是 NVIDIA 官方的训练代码；CPU 上与上游逐位一致（见 [验证记录](docs/VALIDATION.md)），GPU 上的 FSDP、CPU offload 与编译内核需在目标机器确认。
- 与上游的有意差异：checkpoint 只存训练的 adapter 权重、优化器状态和步数，不存整份模型（冻结权重就是已校验的 prepared Base，续训时
  再次校验 Base 与运行设置）；续训严格接着中断处的 batch 序列（HF Trainer 在每个 epoch 的 batch 数不是梯度累积整数倍时会跳到别处）。
- transformers 4.51 加载 Base 时不初始化 checkpoint 里没有的权重（4.57.3 会初始化）。训练引擎按 4.57.3 的规则初始化新增 adapter
  （norm 权重为 1，投影权重随后按官方做法置零），并拒绝 adapter 以外的缺失权重；不这样做时第一步的梯度范数就是 Infinity（已复现）。
- DMS 代码按 transformers 4.57.3 写成，用到了 4.51 没有的 `CacheLayerMixin`/分层 `Cache`、`masking_utils`、`GradientCheckpointingLayer`
  与 `TransformersKwargs`。[`dms/transformers_compat.py`](dms/transformers_compat.py) 在已安装的 transformers 提供这些名字时直接用它的，
  否则用从 4.57.3 原样复制的类或等价的替代。模型类另外显式给出 `config_class` 与 embedding 访问器（4.51 没有默认实现，否则 `lm_head`
  不会与 embedding 绑定）。
- 评测入口在 import 评测框架之前只检查 `lm-eval>=0.4.9`、`datasets>=4.7`（共享缓存中 IFEval 条目的 `Json` 特征类型从 4.7 起可读），
  不满足时报错并提示回到 torch218；精确版本写进每个结果的 `comparison_contract.versions`，`compare_results` 要求双方一致。
- `pyproject.toml` 的核心依赖与 `[eval]` 只有 torch218 满足的下限；不再有训练专用的 extra。
- 此前在另建训练环境（transformers 4.57.3）中导出的 student 仍可直接在 torch218 评测：同一权重在两个版本中导出逐位相同，评测输出
  逐元素相同，用 [`tests/cross_env_check.py`](tests/cross_env_check.py) 核对。
- 若曾在 torch218 中执行过 `pip install -e .` 或安装过旧的 `[train]`/`[eval]`/`[dms-train]` extra，可用
  `python -m pip uninstall semanticlogkv-dms-baseline` 撤销本项目的 editable 条目；旧 extra 若改动过 transformers、datasets、lm-eval 等，
  按环境清单第 1 节导出快照与版本表核对，有差异时交由环境维护者恢复，不在共享环境里自行重装。

**lm-eval 版本决定任务定义。** torch218 的 lm-eval 0.4.11 与 0.4.9 的 LongBench 数据源（`THUDM/LongBench` → `Xnhyacinth/LongBench`）、
prompt 字段（`{{input}}` → `{{question}}`）和解码方式（`do_sample: True` → `False`，即贪心）不同，arc、ceval、ifeval 与 RULER 不变。
只比较同一 lm-eval 版本下的结果；SemanticLogKV 来自其他版本的旧结果须在 torch218 中重跑。

## 准备同一份 Base

在 torch218 中执行（输出是训练的输入；`majob.sh --train` 在它不存在时自动执行这一步）：

```bash
python -m models.qwen3.prepare_base
```

默认读取项目内 `configs/experiment_qwen1.7b_32k.yaml`（SemanticLogKV 实验 `arc_attach_alpha_beta_k12_b128_2k` 继承链的平铺快照，只保留 Base、32K 窗口和 LogKV 对照参数），使用其中 **原始 `ckpt_dir`**（可用 `DMS_BASE_CHECKPOINT` 覆盖），不使用 SemanticLogKV CPT 的输出。`--experiment-config` 只接受项目内文件。输出到 `outputs/semanticlogkv_base`，两份训练配置均指向此处。输出目录已存在时拒绝覆盖。

转换使用项目内 `models/qwen3/convert_lit_to_hf.py`（改编自 LitGPT 0.5.12 的 Qwen3 映射），严格校验参数、绑定 embedding 和模型配置（原始 Base 的 `model_config.yaml` 带有 SemanticLogKV 研究分支的 `use_research`、`research_swa_size`、`research_breakpoint_schedule` 等开关；按评测器的同一规则丢弃 LitGPT Config 没有的全部字段，与取值无关，因为评测器同样忽略它们，被评测的 Base 是普通 dense 结构；丢弃的字段及取值写入 manifest 的 `source_config_ignored_keys`），原样复制 tokenizer，保存来源、实验 YAML、转换器及输出文件 SHA-256。训练启动前验证已转换文件；来源信息随 DMS 模型导出，并在正式评测前与当前比较 Base 再核对。没有 manifest 的旧模型不能直接进入正式对比；早先用相邻 LitGPT 转换的产物（manifest v1）仍按同样哈希校验。

当前配置的权重与测试缓存是服务器 `/home/ma-user/...` 路径，本地 Mac 未存放这些真实文件。应在同样挂载这些目录的 CUDA 机器执行；不会因文件缺失自动下载替代版本。完整说明见 [评测对齐记录](docs/SEMANTICLOGKV_ALIGNMENT.md)。

## 先做小规模检查

不下载权重的 CPU 检查，在 torch218 中、项目根目录运行，不设 `PYTHONPATH`：

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

验证配置、纯文本格式、1.7B 参数尺寸、底模加载与冻结、teacher 数学一致性、适配器梯度、训练循环（设置校验、数据顺序、中途续训
与不中断训练逐位相同、checkpoint 轮换、导出）、保存与重载、评测加载器、启动器与依赖声明。CPU attention 检查使用 eager 执行，
不能证明 CUDA 内核可用；没有 GPU 时跳过 `tests/test_qwen3_cuda.py`。

此前训练环境导出的模型与 torch218 的评测行为可用跨版本检查核对（CPU 即可；DIR 为任意新目录）：

```bash
<transformers 4.57.3 的 python> -m tests.cross_env_check export DIR/student
<transformers 4.57.3 的 python> -m tests.cross_env_check run DIR/student DIR/train.json
<torch218 python> -m tests.cross_env_check run DIR/student DIR/eval.json
<torch218 python> -m tests.cross_env_check compare DIR/train.json DIR/eval.json   # 应输出 IDENTICAL
```

CUDA 检查同样不下载权重，使用随机小模型：蒸馏一步后 prefill/decode（序列 1024，超过 512 的保护窗口），以及按主配置的 FSDP 设置
（CPU offload、模块状态同步、activation checkpointing）训练 3 步、中途续训并导出：

```bash
python -m pytest -q tests/test_qwen3_cuda.py
```

准备上述 Base 后，使用真实 1.7B 权重跑两步，仅读取仓库内合成文本，不下载 OpenR1 数据集：

```bash
bash scripts/train.sh configs/qwen3_1.7b_base_smoke.yaml
```

这是流程检查，**不能用合成数据产出的 checkpoint 评价方法质量**。配置默认单 GPU、1024 tokens、2 步、不启用 FSDP；真实峰值显存需在目标 GPU 上验证。输出为 `outputs/qwen3_1.7b_base_smoke/student_model/`。

## 训练

在 torch218 中执行。只训练：

```bash
# 单机，GPU 数量按机器设置；CUDA_VISIBLE_DEVICES 可选择具体设备
NPROC_PER_NODE=8 bash scripts/train.sh configs/qwen3_1.7b_base.yaml
```

训练后接着评测，一个作业完成（见下文 `majob.sh`）：

```bash
bash majob.sh --train configs/qwen3_1.7b_base.yaml --output outputs/dms-32k-full
```

脚本先由单进程准备数据，再使用 torchrun 启动训练。主配置基于官方 1.7B 调试配方：8K 上下文、544 个优化器步、每卡 batch 1、梯度累积 16、学习率 3e-5、目标 CR 从 1 到 16（第 510 步达到），保留 512-token 延迟窗口。采用 FSDP、CPU offload 和逐层 activation checkpointing；仍需足够的 GPU 与主存，未给出未经实测的显存承诺。OpenR1-Math-220k 第一次准备时需要网络或已有的 Hugging Face 缓存，处理结果缓存在 `dms/training/datasets_cache/`。

训练结束写出 `student_model/training_record.json`：步数、训练设置与完整 YAML、可训练参数量、每步 token 数、设备、torch/transformers/datasets/tokenizers 版本、本次启动用时和最后一条日志；评测把它记入 `evaluated_model.training`。每步日志（loss、grad_norm、学习率、epoch 与 `gl_*` DMS 指标）打印到标准输出，并保存在每个 checkpoint 的 `trainer_state.json`。

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

YAML `trainer.resume_from_checkpoint` 设置为 `auto` 时选择最新的完整 checkpoint（写到一半的 `checkpoint-N.tmp` 不算）；没有 checkpoint 则开始新训练。也可填写明确目录，目录不存在会报错。恢复时须使用同一训练配置、数据与 GPU 数：checkpoint 记录进程数、种子、梯度累积、FSDP 设置、样本数、上下文长度和 Base 权重指纹，不一致时拒绝恢复。

checkpoint 每 `save_steps` 步写一次（最后一步也写），保留 `save_total_limit` 个，内容为 `adapters.pt`（训练的 516,096 个参数，fp32）、`optimizer.pt` 与 `trainer_state.json`。训练结束收集完整 student 权重，由主进程写入 `student_model/`（先写到旁边的临时目录再替换，不残留旧导出的文件）。保存的是完整模型与适配器、tokenizer、模型配置和实现文件；不是只有小型 adapter 文件。

中间训练 checkpoint 可单进程导出：

```bash
python -m models.qwen3.extract \
  --config outputs/qwen3_1.7b_base/config.yaml \
  --checkpoint outputs/qwen3_1.7b_base/checkpoint-34 \
  --output outputs/qwen3_1.7b_base/student-step34
```

该入口读取 checkpoint 的 `adapters.pt`，其余权重来自 `model.name` 指向的已校验 prepared Base；不能读取 HF Trainer 写的旧 checkpoint 或任意外部 sharded checkpoint。

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

模型窗口 32K、batch 1。main 阶段不传 NIAH metadata，与参考脚本一致。删除 `--limit` 才是任务全量。`NPROC_PER_NODE` 控制每卡一份模型，默认 1。启动器私有加载项目内快照的 `eval.py`，在该私有评测实例中接入 DMS 子类，复用 tokenizer、生成循环、评分与指标统计。评测强制离线；map/filter 预处理写进程私有临时文件。共享离线缓存只读：报 `Feature type 'List'/'Json' not found` 说明安装的 datasets 比写缓存的版本旧（环境被降级，例如在旧的 transformers 4.57.3 训练环境中评测），应回到 torch218；某个条目的 `dataset_info.json` 不是合法 JSON 时报错并停止，不移动、不改写缓存条目，见环境清单第 4 节。

### 训练平台作业：majob.sh

`majob.sh` 是平台作业入口，**每个节点执行一次**：先 main（47 个入口）再 NIAH（3 个入口，仅 32768），两阶段各自一次 `torch.distributed.run`，main 失败即返回非零且不启动 NIAH。拒绝 `--suite/--tasks/--evaluation-config/--experiment-config/--print-config`。

- `--model DIR`：评测已导出的模型，不训练。
- `--train TRAIN_YAML`：一体化作业，只在单节点运行。按顺序：prepared Base 不存在时用实验 YAML 准备 → 单进程准备数据 → torchrun 训练（与评测同一解释器与 GPU 数）→ 用这次导出的 `<output_dir>/student_model` 评测 main 与 NIAH。任一步失败即停止，后续阶段不启动；训练没有导出 student 时不会拿旧模型去评测。不能与 `--model` 同时使用。训练与评测仍是分别启动的阶段，只是由同一作业依次执行；不带 `--train` 时脚本从不训练。

```bash
bash majob.sh configs/experiment_qwen1.7b_32k.yaml --model outputs/qwen3_1.7b_base/student_model \
  --limit 4 --output outputs/dms-32k-smoke
bash majob.sh configs/experiment_qwen1.7b_32k.yaml --model outputs/qwen3_1.7b_base/student_model \
  --output outputs/dms-32k-full
# 先用合成数据跑通训练→导出→评测全链路（只证明流程，不评价质量）
bash majob.sh --train configs/qwen3_1.7b_base_smoke.yaml --limit 4 --output outputs/dms-smoke-pipeline
bash majob.sh --train configs/qwen3_1.7b_base.yaml --output outputs/dms-32k-full
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
