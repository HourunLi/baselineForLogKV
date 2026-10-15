# ChunkKV baseline：实现与 SemanticLogKV 对齐口径

更新：2026-10-15。分支：`chunkKV`（自 `DMS` 创建，DMS 代码原样保留）。论文：
[ChunkKV, arXiv:2502.00299v5](https://arxiv.org/html/2502.00299v5)。

ChunkKV 免训练。本项目独立维护实验 YAML 和必要的模型、训练、评测代码，直接读取与
SemanticLogKV CPT 起点相同的 Base `lit_model.pth`，不做 HF 转换。运行代码快照位于
`chunkkv/_vendor/semanticlogkv/`，采用项目内限定导入，不依赖相邻 `litgpt` 安装或另一项目的源码与 YAML。
快照来源、改动与许可证见该目录的 `README.md`、`SOURCE.json`、`LICENSE.md`。
运行环境要求 Python 3.10–3.13、PyTorch >= 2.7；模型推理和训练仍需相应运行依赖与实际权重、数据。

## 项目与配置边界

| 路径 | 用途 |
|---|---|
| `/data/lihourun/baselineForLogKV/chunkKV` | ChunkKV 项目根，以下命令均从此目录执行 |
| `/data/lihourun/semanticLogKV` | 独立的 SemanticLogKV 对照项目，ChunkKV 运行不要求其存在 |
| `configs/chunkkv/experiment_qwen1.7b_32k.yaml` | 自有平铺实验配置，保存 Base、数据、32K 上下文与训练/LogKV 参数，不继承外部 `exp/` YAML |
| `configs/chunkkv/qwen3_1.7b_32k.yaml` | ChunkKV 的选块、预算和层间复用参数 |
| `configs/chunkkv/dense_cpt_qwen1.7b_32k.yaml` | 可选 dense CPT 配置，`experiment_config` 引用同目录的自有实验配置 |
| `outputs/` | 本项目评测与 dense CPT 的默认输出 |

评测用 `--experiment-config` 指定实验配置；旧 `--semantic-config` 只保留为这一参数的别名，
不再定位外部项目。`--litgpt-root` 已移除。实验中的相对资产路径以 ChunkKV 项目根为基准；
dense 配置中的 `experiment_config` 相对其配置文件目录解析。

项目地址与数据地址分别管理：没有证据表明原始权重、训练数据或缓存已迁移，所以仍保留现有
`/home/ma-user/...` 默认位置。迁移这些资产后，分别设置 `CHUNKKV_BASE_CHECKPOINT`、
`CHUNKKV_DATA_DIR`、`CHUNKKV_HF_CACHE`，或修改自有配置。比较 SemanticLogKV 的训练权重时，
用 `--model` 指定实际 checkpoint，不能直接把其项目根当作权重目录。

## 算法与论文的对应

| 论文 | 实现（`chunkkv/core.py`、`chunkkv/model.py`） |
|---|---|
| Alg.1 观察窗打分 | prompt 末尾 `window_size` 个 query 对全部 prompt key 做因果 softmax（模型自身 scale），对窗内 query 与**全部 query head** 求和，每层得到一组 token 分数 |
| Alg.1 块分数 `A_i = Σ_j A_{:,j}` | 窗口之前的前缀从位置 0 起按 `chunk_size` 切块，块分数为块内 token 分数之和；最后不足一块的部分按实际 token 求和 |
| Alg.1 Top-k 与拼接 | 选 `(budget − window) // chunk_size` 个最高分块（稳定排序，分数相同取靠前的块），按原顺序保留原始 K/V，再接上完整观察窗 |
| Alg.2 层间索引复用 | `reuse_layers = N`：仅第 `0, N, 2N, …` 层打分，其后 `N−1` 层复用其索引；`1` 表示不复用 |
| 压缩时机 | prompt 先完整 dense prefill（prompt 自身输出不受影响），每层算完 attention 后压缩一次；之后生成的 token 不再压缩，直接追加 |

同一层所有 KV head 共用一组块（跨 head 求和），与 KVPress `ChunkKVPress` 一致，也是层间索引复用的
前提。保留的是原始 K/V，不是块的均值摘要。

### 需要说明的取值与差异

| 项 | 取值 | 依据 |
|---|---|---|
| `chunk_size` | 10 | 论文默认块长 |
| `window_size` | 64 | 本环境无法访问 arXiv 原文核对观察窗长度；取论文所指 KVPress 实现中 SnapKV 打分器的默认值。可用 `--window-size` 改 |
| 块分数 | 求和、无 pooling | 按 Alg.1；KVPress 的 SnapKV 打分器另有 kernel 5 的 avg-pool，块内求均值，本实现未采用 |
| 观察窗 | 不参与块竞争，单独完整保留一次 | 等价于 Alg.1 "用观察窗替换所选结果的最后 w 个位置"，但不产生重复条目；预算中不足一块的余量（< `chunk_size`）不使用 |
| `budget` | 每层每个 KV head 的 prompt 条目数，含观察窗（论文的 `L_max`） | prompt 不超过预算时不压缩 |
| `reuse_layers` | 默认 1 | 论文主精度结果不依赖复用；复用是加速手段，按需设 2 或 3 |

论文中的其他实现细节（如原始实验代码的 kernel/批处理优化）未复现；计时只反映本实现。

## 与 SemanticLogKV 对齐的评测接口

`scripts/chunkkv_evaluate.sh` → `chunkkv.evaluate` → 私有模块实例中的本地快照 `eval.main`。
`ChunkKVLM` 继承快照中的 `LogKVLM`：构造时调用其构造函数（dense 模式）加载 tokenizer、
`model_config.yaml` 与权重，再把每层 `CausalSelfAttention` 包装为 ChunkKV 缓存；
`_score_tokens / loglikelihood / loglikelihood_rolling / generate_until` 全部直接继承。
同一入口也提供 `--backend logkv`（SemanticLogKV 本身）和 `--backend dense`（同模型 Full KV），
三者的结果 JSON 都带 `comparison_contract`，用 `chunkkv.compare_results` 核对。

| 项目 | 对齐方式 |
|---|---|
| 权重 | 默认本项目实验 YAML 的 `ckpt_dir`（原始 Base），与 CPT 起点相同；`--model` 可换为任意兼容的 LitGPT checkpoint |
| 权重完整性 | 参考加载器是 `strict=False`；本实现另外按 mmap 读取 checkpoint 的键并要求与模型完全一致 |
| tokenizer、任务、长度、样本 | 与 DMS 分支相同：Base 目录 tokenizer；默认 `niah_single_2,niah_single_3`；`block_size` 和 RULER `max_seq_lengths` 都取实验 `context_length=32768`；同 harness、seed、`--limit` |
| 生成与停止 | 共用本地快照的 `litgpt.generate.base.generate`，同截断与 EOS 规则（共享代码不执行 `until` 字符串） |
| 选择题打分 | 参考代码一次前向 context+continuation 并用 `lm_head_start` 标出 context 末位；ChunkKV 以此为 prompt 边界：先 prefill 并压缩 context，再让 continuation 读压缩缓存，等价于逐 token 解码 |
| rolling PPL | 每个窗口只有 1 个 context token，整窗一次 dense 前向，ChunkKV 不改变其结果 |
| KV 统计 | 请求结束、reset 前快照。所选块计为 `exact_slots`，观察窗与之后追加的 token 计为 `recent_slots`，`hierarchical_slots=0`；比例为 Σretained/Σdense |
| 缓存预算记录 | `cache_budget` 写明预算来源、`prompt_budget_tokens`、每 token 字节数、参考 LogKV 字节数；ChunkKV 缓存按请求动态分配，`persistent_cache_bytes` 为 null |
| 计时与显存 | 沿用参考 prefill/decode 计时边界；prefill 时间包含打分与压缩 |
| 来源记录 | `comparison_contract` 含 Base/tokenizer/评测代码/兼容层哈希与依赖版本，另含实际评测权重的 SHA-256（按参考规则解析 `step_*` 后的目录）及 dense 对照的 manifest |

## 预算口径

ChunkKV 是固定预算方法（与 N 无关），SemanticLogKV 是 `O(log N)`。比较时至少报告一种对齐预算：

| 选项 | 含义 | 默认实验下的数值 |
|---|---|---|
| `budget: logkv`（默认） | 先用实验 YAML 的 LogKV 参数构建一次参考缓存，取其已分配持久 buffer 字节数（与 SemanticLogKV 自己 `swa_window_size=0` 的 SWA 对照同一口径），除以 ChunkKV 每 token 的 K/V 字节数 | K=12、B=128、recent=2048、Alpha 256 时为 2,037,252,224 字节 → **17,763 条/层/KV head（32K 的 54%）**，CPU 上用真实 1.7B 配置算得，以实际运行打印为准 |
| `--budget-from-result <logkv结果.json>` | 用 SemanticLogKV 结果中 `generate_until` 请求结束时的平均保留条目数（每层每 KV head）作为 prompt 预算，即逻辑条目数对齐 | 取决于实测；生成 token 另外追加 |
| `--budget N` | 固定条目数 | — |
| `--keep-ratio r` | 每条 prompt 保留 `ceil(r·T)`，论文常用 10% | — |

字节对齐给 ChunkKV 的预算相当宽松（参考 LogKV 的持久分配包含空槽、recent 双缓冲、原始 K 副本等）；
条目对齐更接近论文 "同 KV 条目数" 的比较。建议两者都报，并用 `compare_results` 输出的实际保留比例核对。

## 可选：同数据 dense CPT 对照（训练接口）

ChunkKV 不需要训练，主表行应为 "Base + ChunkKV"。SemanticLogKV 是在 Base 上做 32K prolong CPT 后的
模型，为了区分 "CPT 语料/预算" 与 "压缩注意力" 的贡献，提供一个对照训练入口：

```bash
# 训练环境：兼容的 PyTorch/CUDA 环境，训练依赖由 extra 安装
cd /data/lihourun/baselineForLogKV/chunkKV
python -m pip install -e '.[chunkkv-train]'
python -m chunkkv.train --print-config
NPROC_PER_NODE=8 bash scripts/chunkkv_train.sh configs/chunkkv/dense_cpt_qwen1.7b_32k.yaml
bash scripts/chunkkv_evaluate.sh --model outputs/dense_cpt/qwen1.7b-32k-attach-alpha-beta_k12_b128_2k
```

它私有加载本项目的 `demo.py` 快照，使用同一份自有实验 YAML（Base `ckpt_dir`、数据、batch、步数、优化器、
学习率日程、FSDP 与 checkpoint 格式）；仅把模型类换成子类，使 `enable_log_kv_training` 保持标准
causal attention。固定改动只有：`run_eval: ""`、`save_ckpt: true`、`resume_dir: null`、
`log_kv_semantic_clusters: false`（避免安装路由重放），输出写到本仓库 `outputs/`，并拒绝写入
本地运行快照、原始 Base 或配置中 SemanticLogKV 的 save/tensorboard/staging 目录。`--prepare` 只在首节点运行一次，记录解析后的
配置、YAML 链、训练代码与 Base 权重哈希；之后 torchrun 每个 rank 重新解析并核对未变化才训练。
`overrides` 中改动数据或预算的键会进入 manifest，但会使对照不再等价。

## 运行

```bash
# 评测环境：lm-eval==0.4.9、datasets<4；不安装相邻项目
cd /data/lihourun/baselineForLogKV/chunkKV
python -m pip install -e '.[eval]'

# 只解析本项目配置，不加载权重或训练/评测依赖
python -m chunkkv.evaluate --print-config
python -m chunkkv.train --print-config
# 先用 4 条样本检查三种后端
bash scripts/chunkkv_evaluate.sh --limit 4 --output outputs/cmp-chunkkv
bash scripts/chunkkv_evaluate.sh --backend logkv --model /path/to/semanticlogkv/ckpt --limit 4 --output outputs/cmp-logkv
bash scripts/chunkkv_evaluate.sh --backend dense --limit 4 --output outputs/cmp-dense
# 条目数对齐的 ChunkKV
bash scripts/chunkkv_evaluate.sh --budget-from-result outputs/cmp-logkv/eval_results_TIMESTAMP.json --limit 4 --output outputs/cmp-chunkkv-slots
python -m chunkkv.compare_results outputs/cmp-chunkkv/eval_results_TIMESTAMP.json outputs/cmp-logkv/eval_results_TIMESTAMP.json
```

`NPROC_PER_NODE` 控制每卡一份模型。删除 `--limit` 才是任务全量；`--tasks` 为各方法选同一组任务。
层间复用：`--reuse-layers 2`。在同一 SemanticLogKV CPT 权重上换成 ChunkKV（`--model` 指向它）只是
缓存机制诊断，不是各方法各自适配后的公平比较。

## 验证记录

2026-10-15 独立化改动：在 macOS、Python 3.11 下执行配置与隔离测试，结果为 **20 passed，2 deselected**：

```bash
python -m pytest -q tests/chunkkv/test_standalone.py tests/chunkkv/test_evaluate.py tests/chunkkv/test_train.py \
  -k 'not dense_control_gpt and not private_trainer'
```

覆盖本地 YAML 与环境变量展开、相对路径、输出保护、训练 manifest 和续训检查、三种后端的参数传递，
以及从其他工作目录启动、禁止外部项目与模型库导入时的配置预览。额外核对：平铺配置的 57 个字段与
原配置链展开结果一致；运行快照的语法、内部导入闭包与来源哈希一致；运行快照及许可证纳入打包文件；
两个启动脚本的 Bash 语法通过。完整小模型测试未运行：临时环境的 PyTorch 下载过慢，已停止安装；
这两项被排除的模型/训练器测试和其余模型测试不计入本次通过结果。CUDA 与真实权重验证同样未执行。

以下是 2026-10-13 的历史验证，发生在项目独立化之前，不能作为本次改动的通过记录。
当时环境：Linux CPU，Python 3.13、PyTorch 2.9.1、lightning 2.6.6、lm-eval 0.4.9、datasets 3.6.0、
transformers 4.56.2、litdata 0.2.59；SemanticLogKV 参考为 `HourunLi/LogKV@4fd97c7`，editable 安装核心包。

```bash
python -m pytest -q tests/chunkkv
```

历史结果：**34 passed**。覆盖：

- 打分与选块和逐 head、逐 query 的显式 softmax/循环实现一致；窗口保留、尾块截断、比例预算、
  稳定排序、短 prompt 与参数校验。
- 不压缩时，ChunkKV 包装与 LitGPT 原生 dense KV cache 的 prefill/decode logits 一致（float32，1e-5）。
- 压缩后 decode 与 "屏蔽被丢弃 prompt 位置" 的原生 dense 推理一致；每层用自己的 Q/K 打分、
  复用层沿用上一打分层索引，缓存内容等于 dense K 在所选位置的取值。
- 选择题拆分路径等于 "context prefill + 逐 token 解码"；缓存扩容、输入位置与生命周期检查。
- 通过真实私有加载的 `eval.py`：参考构造函数加载 tiny checkpoint 与 tokenizer，不压缩时
  `generate_until`、`loglikelihood`、`loglikelihood_rolling` 与参考 dense 后端一致；压缩时统计字段正确；
  `logkv` 预算等于参考 LogKV 缓存的持久字节换算；缺键 checkpoint 被拒绝；参考文件与原生类不变。
- 启动器：三种后端共享任务输入，只有 LogKV 或字节对齐时传 LogKV 参数；输出隔离与结果重标注；
  条目对齐预算读取；结果核对拒绝 prompt/运行时不同。实际执行 `--print-config` 四种组合成功。
- dense 对照：解析真实实验 YAML 链后训练超参不变；输出与输入保护；manifest 与续训一致性；
  子类在 demo.py 的调用后 loss 与原生 GPT 一致；私有加载真实 `demo.py` 不改参考文件。

尚未验证：CUDA 上的真实 1.7B 权重推理、RULER 实际数据与得分、多卡评测、显存与速度、
dense 对照的真实训练（`demo.py` 需要 CUDA）。先在目标机器用 `--limit 4` 跑通三种后端并
`compare_results` 核对，再扩大任务。
