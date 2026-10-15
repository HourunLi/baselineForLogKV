# 官方源码与本地修改

来源：[NVIDIA/Model-Optimizer experimental/dms](https://github.com/NVIDIA/Model-Optimizer/tree/f27b47626d0d5886244d0fe1075e52d7d564ddc9/experimental/dms)。固定 commit：`f27b47626d0d5886244d0fe1075e52d7d564ddc9`。核对日期：2026-10-12。

模型尺寸来源：[Qwen/Qwen3-1.7B-Base config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)。代码与模型仍遵守各自原始许可证；已保留文件版权头与根目录 LICENSE。

## 复制范围

| 上游路径 | 当前路径 | 处理 |
|---|---|---|
| experimental/dms/dms | dms | 门控、attention、训练引擎原样保留；data.py 增加空数据报错，cache.py 修正 prefill CR 统计 |
| experimental/dms/models | models | Qwen3 模型和配置原样；训练与导出入口做以下适配 |
| experimental/dms/tests | tests/upstream | 保留原测试，修正独立仓库的测试工具导入 |
| experimental/dms/README.md、ARCHITECTURE.md | docs/upstream | 原文档快照，不代表本仓库命令和验证结果 |
| LICENSE | LICENSE | 原许可证与第三方声明 |

## 其他来源

| 来源 | 当前路径 | 处理 |
|---|---|---|
| SemanticLogKV（`HourunLi/LogKV@4fd97c78e04f0ec96ad6aed18df9e9cf440c6090`）`eval.py` 及其导入闭包（LitGPT 0.5.12 分支） | `models/_vendor/semanticlogkv` | 评测协议私有快照，Apache-2.0；仅改命名空间、头注释与 `DMS_HF_CACHE`；省略 `demo.py`、`data.py`。逐文件来源/打包哈希见其 `SOURCE.json` |
| LitGPT 0.5.12 `litgpt/scripts/convert_lit_checkpoint.py` 的 `copy_weights_qwen_3` | `models/qwen3/convert_lit_to_hf.py` | Apache-2.0；只保留 dense Qwen3 映射，去掉惰性加载 saver，未知键报错 |

## 本地差异

- `configs/qwen3_1.7b_base.yaml`：从官方 1.7B 配置换为 Base、显式纯文本模板、FSDP 按 decoder layer 包装和完整 state dict，关闭外部实验追踪。核心门控、学习率、8K 长度和压缩日程保持原配方。
- `configs/qwen3_1.7b_base_smoke.yaml`：真实 1.7B 模型的两步合成文本试跑，无 FSDP；不用于质量结果。
- `models/qwen3/train.py`：支持本地 `text` JSONL；缺失 resume 路径报错；纯数据准备放在 TrainingArguments 初始化前；向 Trainer 传 tokenizer 使中间 checkpoint 可导出；仅主进程写配置；收集完整 FSDP 权重并单进程导出；导出不原地改写 live shards 的 dtype；保存训练目标 CR 元数据。
- `models/qwen3/extract.py`：checkpoint 张量加载到 CPU，避免按原设备占用 GPU。
- `dms/cache.py`：prefill 时把近期窗口的 contiguous KV 计入 CR；上游仅计 paged 部分，会在尚未开始 decode 时高估压缩率。淘汰和缓存更新算法保持原样。
- `models/qwen3/check_env.py`、`infer.py`、`scripts/`：显式环境检查、Base 续写、训练和可选评测入口。
- `pyproject.toml`：同时打包 `dms` 和 `models`（含私有评测快照），保留官方 Transformers/accelerate 版本，并声明快照所需的 lightning 等依赖；训练 extra 固定 datasets4.4.2，独立评测 extra 使用共享协议的 `lm-eval[ruler,longbench,ifeval]==0.4.9`、datasets<4。
- `prepare_base.py`：读取项目内实验 YAML 的原始 Base，用私有转换器精确转换，复制 tokenizer 并记录文件哈希；训练前验证产物，来源随 DMS config 导出。
- `semanticlogkv_eval.py`、`evaluate.py`、`compare_results.py`、`config.py`：在项目内私有评测快照上接入 DMS，子类继承请求方法，在私有模块实例接入模型/缓存；不修改快照或全局统计函数。任务清单、实验配置、tokenizer、生成/评分与指标均由本项目维护，核对实际样本和比较条件。详细边界见 [评测对齐记录](SEMANTICLOGKV_ALIGNMENT.md)。
- `majob.sh`：训练平台评测入口，每节点执行一次，main 与 32K NIAH 分阶段启动，失败即停止，从不训练。
- `tests/test_qwen3_base.py`、`test_qwen3_cuda.py`：新增本地适配验证；CUDA 测试无设备时明确跳过。

## 不应混淆的口径

官方当前释放的是 adapter-only 训练方案。训练损失源码计算 `KL(student || teacher)` 加压缩目标损失；没有另加标准 CPT 的 LM 训练目标（LM loss 仅记录）。本地保留这一计算，不依据注释的 “forward KL” 标签改变公式。

门控是每层每 KV head 的学习式淘汰，延迟窗口让 token 在真正删除前仍可参与计算；这不是 SemanticLogKV 的显式 meanKV 合并。配置 `dms_cr` 是元数据，不是推理时的硬预算控制。

上游 README 的历史命令和性能数据仅适用于其所述环境，不是本分支已完成的实验。特别是它提及的 `train_small_debug.sh` 与当前源码文件名不一致；本仓库统一使用 `scripts/train.sh`。
