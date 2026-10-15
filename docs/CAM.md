# CaM / Qwen3-1.7B-Base

`CaM` 分支是 CaM 的独立项目（自 DMS 分支创建；DMS 方法代码已移出，在其自己的分支维护）。统一规则见 [AGENTS.md](../AGENTS.md) 与 [Baseline 实现规范](BASELINE_IMPLEMENTATION_GUIDE.md)。

**项目边界**：运行时不读取、不导入相邻 `litgpt`/SemanticLogKV 的源码、实验 YAML 或脚本。SemanticLogKV 评测协议（`eval.py` 及其 LitGPT 依赖）是私有快照 `models/_vendor/semanticlogkv/`（来源 `HourunLi/LogKV@4fd97c7`，Apache-2.0，逐文件哈希见 `SOURCE.json`）；实验配置 `configs/experiment_qwen1.7b_32k.yaml`、任务清单 `configs/eval_suites.yaml`、Base 转换器 `models/qwen3/convert_lit_to_hf.py`（改编自 LitGPT 0.5.12）均在本项目内。共享的只读资产只有原始 Base 权重/tokenizer 与离线评测缓存（`CAM_HF_CACHE` 可改缓存根目录）。

## 方法与边界

来源为 ICML 2024 [CaM 论文](https://proceedings.mlr.press/v235/zhang24n.html)及[作者仓库](https://github.com/zyxxmu/cam)，核对版本 `fc80e64f460b343be96c87596f2494fc9d319c45`。本实现依据其 `generate_task/streaming_llm/kv_cache_cam.py` 与 `pos_shift/modify_llama_cam.py` 的**在线路径**独立适配 Qwen3。没有使用 `cam_hf` 中保留完整 KV、仅加 mask 的实现，也没有复制旧版 Transformers 模型。

设 sink 数 S、recent 数 R、合并接收数 M（官方在线默认 16）。每层每个 token：

1. 追加经 Qwen3 `k_norm` 后、RoPE 前的 K 和 V，按缓存槽位置旋转 K、Q，计算当前 attention 输出。
2. 按共享 KV head 对所属 Q-head 的 attention 概率取平均，并累加到 FP32 分数 A。
3. 长度超过 S+R 后，淘汰索引 e=S。以 `p=clip(A[e]/mean(A[e+1:e+1+M]), 0, 1)` 抽一次 Bernoulli；若选中，则把 `V[e]/M` 加到后续 M 个 V。
4. 删除该 K/V 及对应 A；不合并 K，不转移 A。当前 token 的输出使用合并前状态。

返回缓存至多 S+R 条目，稳态 attention 会临时读取 S+R+1 条目。`recent_size >= merge_size`。零除规则沿用官方：0/0 概率为零，正无穷裁为一。位置使用官方 streaming 路线的 **cache-relative RoPE**；原始位置仍由共享生成接口传入并检查连续性，不能将两种位置政策混为一谈。

Qwen3 适配明确包含：Q/K normalization、共享 GQA 的概率聚合、FP32 累计分数。随机数为每层独立 generator（seed + layer_index），每个请求重置；rank 或投影 chunk 大小不决定抽样次序。以上细节写入结果，实验名称建议 **CaM-online / Qwen3-GQA**，不宣称原论文的原样复现。

## CPT 与权重

CaM 没有可训练参数，**不需要 CPT**。默认对比原始 Qwen3-1.7B-Base + CaM 与同权重 + Full KV；SemanticLogKV 使用其指定 CPT checkpoint。完整系统比较可这样进行，但不能把训练收益全部归因于缓存策略。

`models.qwen3.prepare_base` 从项目实验 YAML 的原始 `ckpt_dir/lit_model.pth`（可用 `CAM_BASE_CHECKPOINT` 覆盖）转换，严格检查 Qwen3 结构、全部权重、tied embedding 和 tokenizer，验证来源、实验 YAML、转换器及转换结果 SHA-256。不会下载另一份同名模型。已有经校验的同一 Base 产物（包括 DMS 分支或早先版本准备的 manifest v1 产物）可以复用，评测前仍按哈希核对。

当前 SemanticLogKV CPT 是 `prolong` 单一语料、32K、全局 batch 32、1200 步；本次没有为免训练 CaM 增加训练语料或另开训练作业。此入口目前只接受有原始 Base manifest 的权重，不能把 SemanticLogKV 压缩 CPT 模型直接当作 CaM 的原始 Base。

## 安装与小样本运行

在项目根目录、已有 BF16 CUDA PyTorch 的环境中执行。依赖沿用共享评测环境：`transformers==4.57.3`、`lm-eval==0.4.9`、`datasets<4`（评测入口启动时检查）。私有评测快照所需的 lightning 等已列入本项目依赖，不再安装相邻 LitGPT。CaM 本身无 FlashAttention/Triton 依赖。

```bash
python -m pip install -e '.[eval,test]'

# 仅首次需要；已有同一 Base 输出目录时跳过此命令，禁止覆盖旧结果
python -m models.qwen3.prepare_base

# 无需加载权重，核对共享任务、tokenizer、cache 路径和参数
python -m models.cam.evaluate --suite main --print-config
python -m models.cam.evaluate --suite niah --print-config

# 推荐第一步：同一任务、长度、样本数下对比 Full KV 和 CaM
bash scripts/evaluate_cam.sh --backend dense --suite niah --limit 4 --output outputs/dense-check
bash scripts/evaluate_cam.sh --backend cam --suite niah --limit 4 --output outputs/cam-check

# 同一入口加载 SemanticLogKV 已训练 checkpoint
bash scripts/evaluate_cam.sh --backend logkv --model /path/to/semanticlogkv/checkpoint \
  --suite niah --limit 4 --output outputs/logkv-check
```

每次必须选择 `--suite main|niah|longbench`（输出进入 `<output>/<suite>/`）或诊断用 `--tasks`。可直接 `torchrun --standalone --nproc_per_node=1 -m models.cam.evaluate ...`。`scripts/evaluate_cam.sh` 默认单卡，可设 `NPROC_PER_NODE=8`，并支持 `NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT` 与 ModelArts 变量。

### 训练平台作业：majob.sh

`majob.sh` 是完整评测入口，**每个节点执行一次**：先 main（47 个入口）再 NIAH（3 个入口，仅 32768），两阶段各自一次 `torch.distributed.run`，main 失败即返回非零且不启动 NIAH。默认评测已准备好的原始 Base + CaM，从不训练、不自动转换 Base；拒绝 `--suite/--tasks/--config/--print-config`。

```bash
bash majob.sh configs/cam_qwen3_1.7b.yaml --limit 4 --output outputs/cam-32k-smoke
bash majob.sh configs/cam_qwen3_1.7b.yaml --output outputs/cam-32k-full
bash majob.sh configs/cam_qwen3_1.7b.yaml --backend dense --output outputs/dense-32k-full
```

平台变量：`MA_NUM_GPUS`（每节点进程数，缺省 `NPROC_PER_NODE`/`LOCAL_WORLD_SIZE`/8）、`MA_NUM_HOSTS`、`VC_TASK_INDEX`、`MA_VJ_NAME`/`MA_TASK_NAME`/`MA_MASTER_INDEX`（主节点 DNS，多节点禁止 localhost）、`MASTER_PORT`（main 用 +1、NIAH 用 +2，可由 `EVAL_MASTER_PORT`/`NIAH_MASTER_PORT` 覆盖）。`CAM_PYTHON` 直接指定解释器，否则激活 `/home/ma-user/anaconda3/bin/activate ${CAM_CONDA_ENV:-torch218}`；环境名不代表实际版本，以结果记录为准。脚本导出 `LITGPT_EXPECTED_WORLD_SIZE`，快照评测器据此检查 rendezvous。

本地开发机没有服务器 `/home/ma-user/...` 下的实际权重、数据缓存或 CUDA。正式实验在有这些挂载的 CUDA 机器运行；缺文件会明确报错。

## 任务与预算

项目任务清单 [eval_suites.yaml](../configs/eval_suites.yaml) 固定参照版本（`HourunLi/LogKV@4fd97c7` 的 `eval.sh`/`majob.sh` 默认值）的 47 个 main 入口：14 个常识/知识/指令任务、20 个 LongBench、13 个 LongBench-E；另有 3 个 NIAH single。按统一协议，NIAH 只评测 `[32768]`，模型窗口 32768、batch 1；main 与 NIAH 分阶段运行，main 阶段不传 NIAH metadata（与参考脚本一致）。参考 shell 原先对 NIAH 默认六档长度，本项目不再使用。

HF 离线缓存路径由本项目设置（`CAM_HF_CACHE`，默认服务器共享缓存），且在 Transformers/Datasets 导入前生效；map/filter 预处理写进程私有临时文件，多 rank 不竞争共享缓存文件；新版 datasets 写入的不兼容缓存条目会明确报错，不升级 datasets、不换数据。RULER metadata 的 `pretrained` 统一为原始 Base 目录字符串，使三种后端共享生成数据和 tokenization 条件；参考 shell 使用各 checkpoint 的 SAVE_DIR 作为这个字段，旧结果不能仅凭任务名直接合并。

继承原有选择题评分、rolling 评分、prompt 左截断、BOS/EOS、生成循环和计时。原评测对 `until` 字符串的处理限制也保持一致。删除 `--limit` 才是任务全量。

默认 `start_size=32,recent_size=2016,merge_size=16`，总容量 2048。可用 `--start-size/--recent-size` 或 YAML 改预算。**这不是与 SemanticLogKV 等字节的承诺**：CaM 的 raw K + V + FP32 A，与 SemanticLogKV 的摘要、原始 K、工作缓冲口径不同，正式等预算比较应根据实际统计调整。

Qwen3-1.7B（28 层、8 KV heads、head_dim 128）在 BF16、2048 条目填满后，CaM K/V 为 224 MiB，FP32 分数为 1.75 MiB，共 225.75 MiB；这是张量公式值，非已测 GPU 峰值，不含模型权重、RoPE、临时拼接和投影/attention 工作区。

## 输出与比较

结果写入 `outputs/` 或显式 `--output`（使用 `--suite` 时再进入 `<suite>/`），包含每次启动一个的 `resolved_eval_<launch_id>.yaml`（不覆盖历史记录）和 `eval_results_*.json`；结果 JSON 内含 `comparison_contract`（含实际评测权重哈希、运行时版本含 pyarrow）与 `resolved_eval`（含解释器、CUDA/驱动、GPU、torchrun 拓扑）。共享入口提供逐样本答案、质量指标、TTFT、prefill/decode 吞吐、请求结束 KV 槽位比例及 CUDA 峰值。CaM compression samples 另有 `kv_storage_bytes`、`attention_score_bytes`、`persistent_cache_bytes`，实测张量存储；不要把槽位压缩倍数当作显存倍数。

```bash
python -m models.qwen3.compare_results \
  outputs/cam-check/niah/eval_results_TIMESTAMP.json \
  outputs/logkv-check/niah/eval_results_TIMESTAMP.json
```

将 TIMESTAMP 替换为实际文件名。检查器验证原始 Base、tokenizer、评测代码/版本、任务/长度和实际样本/prompt/目标/生成参数一致（协议一致性），并单独列出后端、实际评测权重与预算（方法差异）；不要求模型回答相同，也不证明等训练或等内存预算。三方应都从本分支 `models.cam.evaluate` 运行；与其他项目入口的结果相比可能因适配代码指纹不同而拒绝，不能跳过该核对。

当前实现分块计算投影与 MLP，attention 按 token 顺序执行以保证真实在线更新，避免 NxN attention 矩阵。它没有专用 CaM CUDA 内核，32K prefill 会慢于向量化 Full KV；速度是当前实现的测量值，不能据此断言 CaM 算法本身更慢。只在小样本检查有研究价值后再扩大评测。

## 验证

```bash
python -m pytest -q
```

CPU 测试使用真实随机小型 Qwen3 GQA，覆盖公式、概率边界、真实删除、累计分数、原始 K、无淘汰 logits 对齐、chunk/reset 等价、因果性、私有转换器与来源、实际共享选择题/生成链路、离线 cache 初始化、任务清单与 32K NIAH、项目边界、`majob.sh` 两阶段与失败停止及单机/多机参数。完整记录见 [CAM_VALIDATION.md](CAM_VALIDATION.md)。未运行真实 1.7B CUDA 精度、32K 峰值或吞吐实验，不提供虚构结果。
