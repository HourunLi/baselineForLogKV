# CaM baseline（Qwen3-1.7B-Base，SemanticLogKV 评测口径）

`CaM` 分支是 **CaM 在线缓存合并** 的独立项目。CaM 免训练，直接使用与 SemanticLogKV CPT 相同起点的原始 Qwen3-1.7B-Base；适配 Qwen3 的 GQA、Q/K normalization，真实删除 KV，并通过项目内的 SemanticLogKV 评测协议快照复用其请求、评分与指标接口。统一规则见 [AGENTS.md](AGENTS.md) 与 [Baseline 实现规范](docs/BASELINE_IMPLEMENTATION_GUIDE.md)。

**方法、项目边界、安装、三方对比命令和预算口径：[docs/CAM.md](docs/CAM.md)；验证记录：[docs/CAM_VALIDATION.md](docs/CAM_VALIDATION.md)。**

```bash
# 在 SemanticLogKV 的评测环境 torch218 中、项目根目录运行，不安装本项目，也不改动其中任何包
source /home/ma-user/anaconda3/bin/activate torch218
# 已有经校验的 outputs/semanticlogkv_base 时直接复用，无须重新转换
python -m models.qwen3.prepare_base
python -m models.cam.evaluate --suite niah --print-config
bash scripts/evaluate_cam.sh --suite niah --limit 4
# 训练平台完整评测：每节点执行一次，main（47 个入口）→ NIAH（3 个入口，仅 32768），失败即停止
bash majob.sh configs/cam_qwen3_1.7b.yaml --output outputs/cam-32k-full
# 64K / 128K 长文本（流式、同内存）：long（BABILong、长文档困惑度）→ ruler（13 个 RULER 任务）
bash majob.sh configs/cam_qwen3_1.7b_64k.yaml --output outputs/cam-64k
```

- 项目独立：运行时不读取、不导入相邻 `litgpt`/SemanticLogKV 的源码、实验 YAML 或脚本。评测协议快照位于 `models/_vendor/semanticlogkv/`（来源 `HourunLi/LogKV@4fd97c7`，Apache-2.0）；实验配置、任务清单与 Base 转换器均在本项目内。
- 正式评测为 32K：任务清单 [configs/eval_suites.yaml](configs/eval_suites.yaml) 为 47 个 main 入口（14 通用、20 LongBench、13 LongBench-E）与 3 个 NIAH（仅 `[32768]`），main 与 NIAH 分阶段运行。评测强制离线。另有 64K / 128K 长文本评测（配置的 `context:`），CaM 在其中按块合并、与 SemanticLogKV 同内存，见 [docs/CAM.md](docs/CAM.md#64k--128k-长文本流式同内存)。
- 评测在 torch218（lm-eval 0.4.11、datasets 4.8.4、transformers 4.51.3）中进行；2026-11-09 起适配该环境（此前要求 lm-eval 0.4.9、datasets<4、transformers 4.57.3，在 torch218 中启动即失败）。
- 每次评测必须选择 `--suite main|niah|longbench` 或诊断用 `--tasks`；结果按启动编号保存解析配置，并记录实际权重哈希、依赖（含 pyarrow）、解释器、CUDA/驱动与拓扑。
- CaM 不需要 FlashAttention。32K 默认 `merge_interval: 1`，即官方逐 token 在线路径（PyTorch 实现，prefill 较慢）；64K / 128K 每 2048 个 token 合并一次。32K 默认 2048 个条目（`start_size=32`、`recent_size=2016`）是初始预算点，不表示与 SemanticLogKV 等字节。
- 尚未在真实 1.7B 权重、CUDA、服务器离线缓存或多节点作业上运行，没有任何任务分数。
