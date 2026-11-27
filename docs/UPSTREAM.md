# 官方源码与本地修改

来源：[NVIDIA/Model-Optimizer experimental/dms](https://github.com/NVIDIA/Model-Optimizer/tree/f27b47626d0d5886244d0fe1075e52d7d564ddc9/experimental/dms)。固定 commit：`f27b47626d0d5886244d0fe1075e52d7d564ddc9`。核对日期：2026-10-12。

模型尺寸来源：[Qwen/Qwen3-1.7B-Base config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)。代码与模型仍遵守各自原始许可证；已保留文件版权头与根目录 LICENSE。

## 复制范围

| 上游路径 | 当前路径 | 处理 |
|---|---|---|
| experimental/dms/dms | dms | 门控、attention、蒸馏损失与数据管线原样保留；data.py 增加空数据报错，cache.py 修正 prefill CR 统计；cache.py、cache_paged.py 的 transformers 基类改从 `dms/transformers_compat.py` 取；training/engine.py 的修改与新增的 training/loop.py 见下文 |
| experimental/dms/models | models | Qwen3 模型和配置除下文的 transformers 兼容外原样；训练与导出入口做以下适配 |
| experimental/dms/tests | tests/upstream | 保留原测试，修正独立仓库的测试工具导入 |
| experimental/dms/README.md、ARCHITECTURE.md | docs/upstream | 原文档快照，不代表本仓库命令和验证结果 |
| LICENSE | LICENSE | 原许可证与第三方声明 |

## 其他来源

| 来源 | 当前路径 | 处理 |
|---|---|---|
| SemanticLogKV（`HourunLi/LogKV@4fd97c78e04f0ec96ad6aed18df9e9cf440c6090`）`eval.py` 及其导入闭包（LitGPT 0.5.12 分支） | `models/_vendor/semanticlogkv` | 评测协议私有快照，Apache-2.0；仅改命名空间、头注释与 `DMS_HF_CACHE`；省略 `demo.py`、`data.py`。逐文件来源/打包哈希见其 `SOURCE.json` |
| LitGPT 0.5.12 `litgpt/scripts/convert_lit_checkpoint.py` 的 `copy_weights_qwen_3` | `models/qwen3/convert_lit_to_hf.py` | Apache-2.0；只保留 dense Qwen3 映射，去掉惰性加载 saver，未知键报错 |

## 其他来源（transformers）

| 来源 | 当前路径 | 处理 |
|---|---|---|
| transformers 4.57.3 `src/transformers/cache_utils.py` 的 `CacheLayerMixin`、`Cache` | `dms/transformers_compat.py` | Apache-2.0；类体原样复制（`Cache` 的 docstring 缩短），只在已安装的 transformers 没有 `CacheLayerMixin` 时使用；此时 `Cache` 继承已安装的旧 `Cache`，只为让 transformers 自己的 `generate()` 接受 DMS cache |

## 本地差异

- `dms/transformers_compat.py`（新增）：DMS 按 transformers 4.57.3 写成，评测环境 torch218 是 4.51.3。每个名字优先取已安装的 transformers：
  `CacheLayerMixin`/`Cache` 缺失时用上表复制的类（两者成对使用；复制的 `Cache` 继承 4.51 的 `Cache` 只为通过 `generate()` 的 isinstance
  检查，DMS 用到的成员全部来自 4.57.3）；`create_causal_mask` 缺失时对没有 padding 的输入（没有 mask，或 `generate()` 与训练 collator 传入的
  全 1 二维 mask）返回 `None`，与 4.57.3 在 SDPA 下的返回值相同（DMS 训练不读 mask 的值，推理只把最后一行当作 key padding 掩码），遇到 padding 报错；`GradientCheckpointingLayer` 缺失时为普通模块
  （启用 HF 梯度检查点即报错；训练用 FSDP 的逐层 activation checkpointing）；`TransformersKwargs` 为同名 TypedDict；通用任务头缺失时为 `None`。在 4.57.3 中运行时用的就是上游原类。
- `models/qwen3/modeling_qwen3_dms.py`：上述名字从兼容模块导入；`Qwen3PreTrainedModelDMS` 显式写 `config_class`（4.51 不从类型注解推断）；
  `Qwen3ModelDMS`、`Qwen3ForCausalLMDMS` 显式给出 embedding 访问器（4.51 没有默认实现，`lm_head` 不会绑定，加载时报缺 `lm_head.weight`），
  返回值与 4.57.3 的默认实现相同；`_supports_cache_class = True`（4.51 的 `generate()` 据此接受 `DMSCache`，4.57.3 不读取）；
  序列分类、token 分类与问答头只在 transformers 提供通用头时定义。
- `models/qwen3/configuration_qwen3_dms.py`：transformers 没有 `layer_types` 时按 4.57.3 `Qwen3Config` 的规则推出。
- `models/qwen3/semanticlogkv_eval.py`、`infer.py`：`from_pretrained` 用两版都接受的 `torch_dtype=`（4.56 起的 `dtype=` 在 4.51 中会被当成配置字段）。

- `dms/training/loop.py`（新增，非 NVIDIA 代码）：代替 HF Trainer + accelerate 的训练循环，使训练也在 torch218（没有 accelerate）中运行。
  对配方用到的设置重现 transformers 4.57.3 Trainer + accelerate 1.4.0：每个 epoch 用 `seed + epoch` 播种的 `randperm`（accelerate 的
  SeedableRandomSampler），多进程按位置轮流分配并从头补齐（BatchSamplerShard、even_batches）；每个 epoch 按梯度累积分组、末组可以更短，
  loss 除以组大小，组内最后一个 batch 才同步梯度；`max_grad_norm` 裁剪，fused AdamW（4.57.3 在 torch>=2.8 的默认），常数学习率；
  `global_step` 与组内位置喂给 DMS 噪声种子和压缩计划（原 DMSGradAccCallback 的时点）；bf16 autocast；设 `fsdp` 时用 torch FSDP，
  参数与 accelerate 由同一 YAML 得出的一致（bf16 的 MixedPrecision 含缓冲、CPU offload、按 decoder layer 自动包装、use_orig_params、
  sync_module_states、不预取、limit_all_gathers，逐层非重入 activation checkpointing），否则多进程用 DDP（`find_unused_parameters=True`，
  同 Trainer 对非 PreTrainedModel 的设置）。日志与 ModifiedTrainer 相同（`gl_*` 为各进程最后一个 batch 的平均）。
  有意差异：checkpoint 只存 adapter、优化器状态与步数（`adapters.pt`、`optimizer.pt`、`trainer_state.json`），续训校验运行设置与 Base
  指纹，并严格接着中断处的 batch 序列；未实现的 TrainingArguments 取值直接报错。
- `dms/training/engine.py`：`from_pretrained` 用 `torch_dtype=`；transformers 4.51 不初始化 checkpoint 中缺失的权重，学生模型加载后按
  4.57.3 `_init_weights` 的规则初始化新增 adapter（RMSNorm 权重 1、Linear 正态、bias 0），其他缺失、多余或形状不符的权重报错；
  删除 `ModifiedTrainer` 与 `DMSGradAccCallback`（职责移入 loop.py），`DMSTrainerState` 接口不变。
- `configs/qwen3_1.7b_base.yaml`：从官方 1.7B 配置换为 Base、显式纯文本模板、FSDP 按 decoder layer 包装和完整 state dict，关闭外部实验追踪。核心门控、学习率、8K 长度和压缩日程保持原配方。训练设置段由 `hf_trainer` 改名为 `trainer`（字段名与含义不变，去掉只影响 HF checkpoint 格式的 `save_safetensors`）。
- `configs/qwen3_1.7b_base_smoke.yaml`：真实 1.7B 模型的两步合成文本试跑，无 FSDP；不用于质量结果。
- `models/qwen3/train.py`：支持本地 `text` JSONL；缺失 resume 路径报错；训练设置先于数据准备校验；用 `dms.training.loop` 训练；仅主进程写配置；收集完整 FSDP 权重并单进程导出（先写临时目录再替换），附 `training_record.json`；导出不原地改写 live shards 的 dtype；保存训练目标 CR 元数据。
- `models/qwen3/extract.py`：读取 loop.py checkpoint 的 `adapters.pt`，其余权重来自已校验的 prepared Base；tokenizer 取自 Base。
- `dms/cache.py`：prefill 时把近期窗口的 contiguous KV 计入 CR；上游仅计 paged 部分，会在尚未开始 decode 时高估压缩率。淘汰和缓存更新算法保持原样。
- `models/qwen3/check_env.py`、`infer.py`、`scripts/`：显式环境检查、Base 续写、训练和可选评测入口。训练与评测都只要求 transformers>=4.51 及实际用到的 CUDA、FlexAttention 接口，评测另需 FlashAttention。`scripts/train.sh`、`scripts/evaluate.sh` 与 `majob.sh` 设置 SemanticLogKV 的 CUDA 12.8 库路径。
- `pyproject.toml`：同时打包 `dms` 和 `models`（含私有评测快照）。核心依赖只写 torch218 满足的下限（`transformers>=4.51` 与快照所需的 lightning 等），训练数据管线所需的 `datasets>=4.7`、numpy 也只写下限；不声明 accelerate、sentencepiece，也不再有上游固定版本的训练 extra；`[eval]` 为 `lm-eval[ruler,longbench,ifeval]>=0.4.9`、`datasets>=4.7`、pandas、xlsxwriter，只用于另建环境。
- `prepare_base.py`：读取项目内实验 YAML 的原始 Base，用私有转换器精确转换，复制 tokenizer 并记录文件哈希；训练前验证产物，来源随 DMS config 导出。源 `model_config.yaml` 中 LitGPT Config 没有的字段（SemanticLogKV 的 research 开关）按评测器的规则全部丢弃并记入 manifest。
- `semanticlogkv_eval.py`、`evaluate.py`、`compare_results.py`、`config.py`：在项目内私有评测快照上接入 DMS，子类继承请求方法，在私有模块实例接入模型/缓存；不修改快照或全局统计函数。任务清单、实验配置、tokenizer、生成/评分与指标均由本项目维护，核对实际样本和比较条件。详细边界见 [评测对齐记录](SEMANTICLOGKV_ALIGNMENT.md)。
- `majob.sh`：训练平台入口，每节点执行一次，main 与 32K NIAH 分阶段启动，失败即停止；默认取可见卡数，启动前做 CUDA 自检，单节点用 `--standalone`。只有显式 `--train` 时（单节点）先准备 Base 与数据并训练，再评测本次导出的 student。
- `tests/test_qwen3_base.py`、`test_qwen3_cuda.py`、`test_training_loop.py`：新增本地适配验证；CUDA 测试无设备时明确跳过。`tests/cross_env_check.py`：在一个 transformers 版本中导出、在另一个版本中运行评测接口并逐元素比较。`tests/test_transformers_compat.py`：`DMSCache` 被已安装的 transformers 认作 `Cache`，旧版 mask 替代接受未 padding 的输入、拒绝 padding。

## 不应混淆的口径

官方当前释放的是 adapter-only 训练方案。训练损失源码计算 `KL(student || teacher)` 加压缩目标损失；没有另加标准 CPT 的 LM 训练目标（LM loss 仅记录）。本地保留这一计算，不依据注释的 “forward KL” 标签改变公式。

门控是每层每 KV head 的学习式淘汰，延迟窗口让 token 在真正删除前仍可参与计算；这不是 SemanticLogKV 的显式 meanKV 合并。配置 `dms_cr` 是元数据，不是推理时的硬预算控制。

上游 README 的历史命令和性能数据仅适用于其所述环境，不是本分支已完成的实验。特别是它提及的 `train_small_debug.sh` 与当前源码文件名不一致；本仓库统一使用 `scripts/train.sh`。
