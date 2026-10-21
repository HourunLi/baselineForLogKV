# Qwen3 Hybrid CSA4 baseline（`codex/hybrid-csa4`）

从原始 **Qwen3-1.7B-Base** 出发，冻结原参数，移植 DeepSeek CSA4 的学习式重叠压缩和 indexer，采用 **`[Full, CSA, CSA] × 9 + Full`**：10 个 Full 层、18 个 CSA 层。保持 Qwen 的 GQA、独立 K/V 和 RoPE，因此是机制移植，不是 DeepSeek V4 整模复现。

默认局部窗口 128、压缩项 top-k 512。32K 时主 KV 理论保留 **52.0368%**，即 **1.9217× 压缩、节省 47.9632%**；另计 index key 和压缩器状态后为 **52.5548%** 元素保留率。只报告理论元素，不用实际显存衡量压缩。

实现、语料准备、训练、配置、评测和平台启动器均在本项目内。无运行时跨 baseline 源码依赖。采用固定 47 个 main 入口与 3 个 NIAH，仅 32768 上限；比较双方使用同一 torch218 评测环境，不向该环境安装包。**训练也在 torch218 中运行**：训练只用 PyTorch、LitData、PyArrow，torch218 都已具备，所以一个 `majob.sh` 用同一个解释器完成准备、训练和评测。详见 [实现与运行说明](docs/HYBRID_CSA.md)、[统一规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md) 与 [环境清单](docs/BASELINE_ENVIRONMENT.md)。

```bash
# 无模型、无 GPU 也可检查配置和预算
python -m hybrid_csa.budget --tokens 32768
python -m hybrid_csa.train --print-config
python -m hybrid_csa.evaluate --suite niah --print-config

# 训练与评测都用已有 torch218（缺省即此路径）；不再设置单独的训练解释器
export HYBRID_CSA_PYTHON=/home/ma-user/anaconda3/envs/torch218/bin/python
bash majob.sh --print-config

# 平台每节点执行一次：准备数据 → 三阶段适配 → main → NIAH，不需要任何参数。
# 输出写到 configs/hybrid_csa/majob.yaml 的 output：/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4
# 每次运行覆盖它：环境预检通过后先删除上次的 train/、eval/、main/、niah/ 与运行记录，其他文件不动；
# 其中的 tokenized/ 是第一次运行准备的分词数据，之后的运行校验后直接复用
bash majob.sh
# --output 可改用别的运行目录（同样覆盖；多节点使用同一共享绝对路径）

# 已有产物时只评测；--limit 仅限制评测样本，不缩短训练
bash majob.sh --mode eval --adapter /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4/train/sparse.pt \
  --limit 4 --output /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/hybrid-csa4-eval-smoke

# 本地实现验证（需项目基础依赖和 pytest）
python -m pytest -q tests/hybrid_csa
```

默认全流程约 524 万训练 token exposures。评测部分的命令、参数、输出位置与环境变量和此前完全相同。适配分为压缩器 LM → indexer KL → sparse 联合训练，不使用捞针训练数据，不按捞针结果调布局或挑 checkpoint。一体化入口校验本次 `train/sparse.pt` 后交给评测；单独评测显式传入 adapter。任一阶段失败即停止，记录实际 Base、tokenizer、数据、代码、环境和训练成本。

本机仅完成小模型与入口验证；尚未执行原始 1.7B 的适配或 32K GPU 评测，没有捞针分数，也不保证初始训练预算足够。代码为纯 PyTorch 基线，未移植 DeepSeek 专用加速内核。

原 ChunkKV 代码保留，说明见 [CHUNKKV.md](docs/CHUNKKV.md)，其平台入口为 `scripts/chunkkv_majob.sh`；根目录 `majob.sh` 已切换为本分支 HybridCSA。
