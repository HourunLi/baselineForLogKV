# ChunkKV baseline（`chunkKV` 分支）

本分支是 [ChunkKV](https://arxiv.org/html/2502.00299v5) 的独立项目，作为 SemanticLogKV 的对比 baseline（DMS 在 `DMS` 分支单独维护，本项目不含其代码）。统一规则见 [AGENTS.md](AGENTS.md) 与 [Baseline 实现规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md)。ChunkKV 免训练：prompt 先完整 prefill，再按观察窗注意力选出整块（默认 10 token）原始 K/V 加观察窗保留，生成 token 直接追加；支持论文的层间索引复用。**ChunkKV 的配置、模型和训练/评测运行代码均在本项目内，不需要相邻 `litgpt` 或 SemanticLogKV checkout。**

- **配置与模型**：自有平铺实验配置在 `configs/chunkkv/experiment_qwen1.7b_32k.yaml`，直接加载其中原始 Base 的 `lit_model.pth`（兼容 LitGPT 权重格式，不转换），即 CPT 的同一起点。
- **评测**：本地运行快照在 `chunkkv/_vendor/semanticlogkv/`，`ChunkKVLM` 继承其中 `LogKVLM` 的请求、生成、打分与 KV/速度指标；同一入口还可跑 `--backend logkv`（SemanticLogKV）与 `--backend dense`（Full KV），结果带 `comparison_contract`。`scripts/chunkkv_evaluate_all.sh` 按 Semantic 当前默认脚本分两阶段跑：47 个主任务入口（含 33 个 LongBench/LongBench-E）及三种 NIAH；按本次实验要求仅评测 32K；任务配置独立保存在本项目内。
- **预算**：默认按参考 LogKV 缓存的持久分配字节对齐（同其 SWA 对照口径）；也可用 `--budget-from-result` 按 SemanticLogKV 实测保留条目数对齐，或 `--budget` / `--keep-ratio`。
- **训练接口（可选对照）**：ChunkKV 本身不训练；`scripts/chunkkv_train.sh` 用本地训练运行快照和同一份自有实验配置做同数据、同预算的 dense CPT，供 "CPT + ChunkKV" 对照。

服务器项目根为 `/data/lihourun/baselineForLogKV/chunkKV`。SemanticLogKV 项目位于 `/data/lihourun/semanticLogKV`，仅作为独立对照项目；比较其训练权重时用 `--model` 指定实际 checkpoint。原始权重、训练数据和缓存的默认位置仍保留现有 `/home/ma-user/...`，可分别通过 `CHUNKKV_BASE_CHECKPOINT`、`CHUNKKV_DATA_DIR`、`CHUNKKV_HF_CACHE` 覆盖。
ChunkKV 使用 Python 3.10–3.13、PyTorch >= 2.7。

```bash
cd /data/lihourun/baselineForLogKV/chunkKV
# 在 SemanticLogKV 的评测环境 torch218 中只登记本项目，不改动其 lm-eval 0.4.9 / datasets 4.8.4 等依赖
python -m pip install --no-deps -e .
python -m chunkkv.evaluate --suite main --print-config
python -m chunkkv.evaluate --suite niah --print-config
python -m chunkkv.train --print-config
bash scripts/chunkkv_evaluate_all.sh --limit 4 --output outputs/cmp-chunkkv
bash scripts/chunkkv_evaluate_all.sh --backend logkv --model /path/to/semanticlogkv/ckpt --limit 4 --output outputs/cmp-logkv
# 比较各阶段本次生成的结果文件；将 TIMESTAMP 替换为实际文件名中的时间戳
python -m chunkkv.compare_results outputs/cmp-chunkkv/main/eval_results_TIMESTAMP.json outputs/cmp-logkv/main/eval_results_TIMESTAMP.json
python -m pytest -q tests/chunkkv
```

训练平台作业使用根目录 `majob.sh`，沿用 Semantic 的 `MA_*` 节点约定，默认激活 `torch218`，
直接执行 Base + ChunkKV 的完整 32K 评测：

```bash
bash majob.sh configs/chunkkv/experiment_qwen1.7b_32k.yaml --output outputs/chunkkv-base-32k-full
```

每个节点执行一次，先 main（47 个入口）再 NIAH（3 个入口，仅 32768），main 失败即停止；`chunkkv.evaluate` 必须显式给出
`--suite` 或 `--tasks`。评测环境为 SemanticLogKV 的 torch218（`lm-eval==0.4.9`、`datasets==4.8.4`，启动时核对），
不要向其中安装本项目的 extra；可用 `python -m chunkkv.check_dataset_cache` 检查共享离线缓存是否都能被该版本读取。环境选择、多节点参数、实现细节、
预算口径与验证记录见 [docs/CHUNKKV.md](docs/CHUNKKV.md)。尚未在 CUDA 上用真实权重运行。

