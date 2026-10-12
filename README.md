# Qwen3-1.7B-Base DMS baseline

在 `DMS` 分支接入 NVIDIA 官方 **Dynamic Memory Sparsification**，底模为 SemanticLogKV 当前实验所用的原始 Qwen3-1.7B-Base。直接从相邻 `litgpt` 仓库配置指定的 `ckpt_dir/lit_model.pth` 转换，不重新下载同名模型。冻结底模，仅训练每层、每个 KV head 的淘汰适配器；同一底模关闭淘汰后作为蒸馏 teacher。

这是一份可训练的实现，仓库不包含已经训练好的 1.7B DMS 权重。采用的是官方当前的 **adapter-only** 配方，不能表述为原论文全参数训练结果的复现，也不是 KVPress 的 `DMSPress` 包装器。

**仓库边界：所有 baseline 接入代码只放在本仓库。** `litgpt` 是只读参考实现，不添加 DMS 分支、注册表或指标钩子。DMS 在本仓库继承其现有评测类、适配模型与缓存接口；评测输出也保存在参考仓库之外。后续新增 baseline 继续遵守这一边界。

## 实现与来源

- 官方门控、延迟淘汰、FlexAttention 训练、压缩 prefill 和 paged KV decode 源码保留在 `dms/`；修正了 prefill 阶段漏计近期窗口的 CR 统计。
- Qwen3 模型、训练、导出和推理入口位于 `models/qwen3/`。结构从 Base 配置读取，无硬编码 8B 层数。
- 1.7B Base：28 层、hidden 2048、16 个 Q heads、8 个 KV heads、head_dim 128、绑定 embedding。新增可训练参数为 **516,096**。
- 官方源码固定于 `f27b47626d0d5886244d0fe1075e52d7d564ddc9`；来源与修改记录见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。许可证见 [LICENSE](LICENSE)。

## 安装

推荐使用上游的 CUDA 环境 `nvcr.io/nvidia/pytorch:25.11-py3`，Python 3.10–3.13。训练需要支持 BF16 的 NVIDIA GPU、PyTorch FlexAttention/Triton；推理另外需要匹配 PyTorch/CUDA 的 FlashAttention，且包含 `flash_attn_with_kvcache`。当前实现不支持昇腾 NPU、MPS 或 CPU 的正式训练/推理。

在已有 PyTorch/CUDA 环境中执行：

```bash
python -m pip install -e '.[train,test]'
python -m pip install -e ../litgpt
python -m models.qwen3.check_env
```

训练环境固定 `transformers==4.57.3`、`datasets==4.4.2`、`accelerate==1.4.0`。评测需单独环境：共享 SemanticLogKV 的 `lm-eval==0.4.9`、`datasets<4`，不能同时安装 `[train,eval]`。LitGPT 只安装核心包，不安装会与 DMS Transformers 版本冲突的 `[extra]`。先准备匹配的 PyTorch/CUDA 与 FlashAttention，再检查依赖解析结果。预检实际检查 `AuxRequest`、CUDA、BF16 和 FlashAttention 接口；只有训练时可使用 `--training-only`。

## 准备同一份 Base

```bash
python -m models.qwen3.prepare_base
```

默认读取 `../litgpt/exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml` 及其继承配置，使用其中 **原始 `ckpt_dir`**，不使用 SemanticLogKV CPT 的 `save_path`。输出到 `outputs/semanticlogkv_base`，两份训练配置均指向此处。输出目录已存在时拒绝覆盖。

转换复用该 LitGPT 仓库的 Qwen3 权重映射，严格校验参数、绑定 embedding 和模型配置，原样复制 tokenizer，保存来源及输出文件 SHA-256。训练启动前验证已转换文件；来源信息随 DMS 模型导出，并在正式评测前与当前比较 Base 再核对。没有 manifest 的旧模型不能直接进入正式对比。

当前配置的权重与测试缓存是服务器 `/home/ma-user/...` 路径，本地 Mac 未存放这些真实文件。应在同样挂载这些目录的 CUDA 机器执行；不会因文件缺失自动下载替代版本。完整说明见 [评测对齐记录](docs/SEMANTICLOGKV_ALIGNMENT.md)。

## 先做小规模检查

不下载权重的 CPU 检查：

```bash
python -m pytest -q tests/test_qwen3_base.py tests/upstream/test_dms_utils.py
```

验证配置、纯文本格式、1.7B 参数尺寸、底模加载与冻结、teacher 数学一致性、适配器梯度、保存与重载。CPU attention 检查使用 eager 执行，不能证明 CUDA 内核可用。

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

正式横向评测在单独的评测环境中执行，两种方法使用同一入口：

```bash
python -m pip install -e '.[eval]'
python -m pip install -e ../litgpt
bash scripts/evaluate.sh outputs/qwen3_1.7b_base/student_model --limit 4
# 使用同一配置、任务、样本数检查 SemanticLogKV CPT checkpoint
bash scripts/evaluate.sh /path/to/semanticlogkv/checkpoint --backend logkv --limit 4
```

默认 `niah_single_2,niah_single_3`、32K、batch 1，来自当前 Alpha/Beta 实验口径。删除 `--limit 4` 才是任务全量；`--tasks` 可为双方选同一组其他任务。`NPROC_PER_NODE` 控制每卡一份模型，默认 1。baseline 启动器只读加载原版 `../litgpt/eval.py`，在自身私有评测实例中接入 DMS 子类，复用 tokenizer、生成循环、评分与指标统计。无需修改或注册任何代码到 `litgpt`。

结果包含 `comparison_contract` 和 harness 逐样本日志，可自动核对：

```bash
python -m models.qwen3.compare_results /path/to/dms_result.json /path/to/semantic_result.json
```

只有 Base 来源、tokenizer、评测代码、运行时版本、任务设置及实际样本/prompt/目标/生成参数一致才通过。旧评测结果若缺这些记录，需要先补小样本对照，不能直接宣称等价。

## 论文实验口径

本实现名称建议记录为 **DMS-adapter / Qwen3-1.7B-Base**。记录底模版本、语料、训练 token 数、适配器参数量、目标与实测 CR、上下文长度。与 SemanticLogKV 比较时，需要另外明确语料和训练预算是否一致；当前默认 OpenR1/8K 配方本身不构成公平性证明。

当前交付包含实现、共享评测接口和离线验证，不包含正式训练结果。检查范围与环境限制见 [验证记录](docs/VALIDATION.md)。CPU 检查不覆盖 CUDA 编译、FlashAttention decode、FSDP 多卡执行或真实 1.7B 精度。
