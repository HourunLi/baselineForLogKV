# 验证记录

只记录本分支实际执行过的检查。配置检查、CPU 局部测试与真实模型/全量评测分开表述。

## 2026-11-11：建立 SnapKV 分支（仅 32K）

### 来源与改动范围

- 评测器、模型、cache、启动器、任务清单与测试框架取自 `PyramidKV` 分支 97b4337（加入 64K/128K 之前、已对齐 torch218 的 32K 版本），
  包名 `pyramidkv` → `snapkv`、环境变量 `PYRAMIDKV_*` → `SNAPKV_*`、配置 `qwen3_1.7b_32k_snapkv.yaml`。按改名还原后逐文件比较，
  以下文件与该版本**逐字相同**：`compare_results.py`、`data_env.py`、`dist_utils.py`、`evaluate.py`、`generate.py`、`kv_cache.py`、`lm.py`、
  `metrics.py`、`qwen3.py`、`qwen3_config.py`、`tokenizer.py`、`majob.sh`、`scripts/*.sh`、`pyproject.toml`、`configs/eval_suites.yaml`。
  所以请求语义、截断、生成、打分、KV/速度指标、结果格式、离线缓存处理和平台启动与 PyramidKV/H2O 分支一致。
- 只改方法相关部分：`snapkv/options.py`（各层同一预算，去掉金字塔与 `beta`）、`snapkv/compressor.py`（来源注释；打分与 top-k 代码与 PyramidKV
  分支相同）、`snapkv/method.py`（去掉 `--beta`）、配置文件，以及对应测试。
- `AGENTS.md`、`CLAUDE.md`、`docs/BASELINE_IMPLEMENTATION_GUIDE.md`、`docs/BASELINE_ENVIRONMENT.md` 与 `DMS` 分支（仓库默认分支）当前版本、
  PyramidKV 97b4337 逐字相同（32K 版本），未修改。
- 上游：KVCache-Factory `68cd9551a63fedf362ddb0edb008badd04c24996`（`pyramidkv/pyramidkv_utils.py` SHA-256
  `a8a1e813…37ce4`，`run_longbench.py` `804fb707…1a7b`）；FasterDecoding/SnapKV `e216ddc84c5bd210378cbdbbba12ba02102aa640`
  （`snapkv/monkeypatch/snapkv_utils.py` SHA-256 `d7eddeaf…4cf5`）。只在开发时读取，运行时不访问。

### 环境

开发容器（CPU，无 GPU）中的 torch218 Python 副本：Python 3.10.20、torch 2.11.0（PyPI 默认 CUDA 版，只在 CPU 上执行）、lm-eval 0.4.11、
datasets 4.8.4、transformers 4.51.3、tokenizers 0.21.4、huggingface-hub 0.36.2、evaluate 0.4.6、fsspec 2023.10.0、pyarrow 23.0.1、numpy 2.2.6、
pandas 2.3.3、xlsxwriter 3.2.9、safetensors 0.7.0、PyYAML 6.0.3、nltk 3.9.4、pytest 9.1.1；与 torch218 一样没有 accelerate、openpyxl、
sentencepiece。副本按环境清单第 1 节的版本表逐个安装（没有完整 freeze），只能证明 Python 层兼容。

### 检查结果

| 检查 | 结果 |
|---|---|
| `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -m pytest -q`（副本） | **80 passed**，无 skip |
| 保留条目与上游一致（`tests/test_compressor.py`） | `query_head` 对照 FasterDecoding/SnapKV `update_kv` 原文、`kv_head` 对照 KVCache-Factory `_grouped_window_attn_cache`/`_select_topk_kv` 原文；maxpool/avgpool、mean/max/sum、三组预算/窗口/核长，各层保留的位置集合完全相同、K/V 逐元素相同（本项目按位置排序存储），在保留条目上的 attention 输出一致 |
| 与独立实现的端到端对照（`tests/test_upstream_pipeline.py`） | Transformers 4.51.3 `Qwen3ForCausalLM` dense prefill → 每层用上游 `kv_head` SnapKV 替换 cache → 在真实位置逐 token decode，与本项目模型 + SnapKV cache 的 prefill/decode logits 在 1e-4 内一致；decode logits 与 dense 明显不同（压缩确实生效） |
| 真实形状单层对照（一次性脚本，未入测试） | Qwen3-1.7B 的 16 Q / 8 KV head、head_dim 128、32512 token prompt（32768 减 256 生成预留），bf16 与 fp32：保留 2048 条（2040 选中 + 8 窗口），`kv_head` 与 KVCache-Factory、`query_head` 与 FasterDecoding 原文的保留集合完全相同；CPU 上单层压缩约 0.25–0.3 s |
| 各层预算 | 28 层、预算 2048 时任一 ≥2048 token 的 prompt 每层都保留 2048 条；短于预算时整段保留；prompt 恰等于预算时不删除任何条目，后续 logits 与 dense 一致 |
| `--print-config` 不依赖 torch | 在没有 torch 的 Python 3.13 中 `--suite main/niah --print-config` 正常输出；给一份 Qwen3-1.7B 形状的 `config.json`（28 层）时 `entries_per_layer_for_full_window_prompt` 为 28 个 2048，`--max-capacity-prompt 17763` 时为 28 个 17763；`--max-capacity-prompt 8`（不大于窗口）在加载任何东西之前报错。测试另在禁止 import torch/lm_eval/datasets/transformers/tokenizers/litgpt 等的情况下运行，并检查只读项目内 YAML |
| `scripts/evaluate.sh configs/qwen3_1.7b_32k_snapkv.yaml --suite niah --print-config`（副本解释器，从项目外目录调用） | 经 `python -m torch.distributed.run --standalone` 正常输出 |
| 任务名解析：`configs/eval_suites.yaml` 的 47 个 main 与 3 个 NIAH 交给 lm-eval 0.4.11 `TaskManager` | 50 个全部可解析，`mmlu`、`ceval-valid` 为任务组；LongBench 任务文件为 `Xnhyacinth/LongBench`、`{{question}}`、`do_sample: false`。只检查注册与任务文件，没有读取数据 |
| `majob.sh` 模拟解释器测试 | 无平台变量时取可见卡数；请求数超过可见卡数、没有可见设备或 CUDA 初始化失败时在 torchrun 前退出；`LD_LIBRARY_PATH` 以 `/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/lib64` 开头；单节点用 `--standalone`；多节点的 rank、主节点 DNS、6001/6002 端口；main 失败即停止；拒绝协议参数与多节点 localhost |
| 经 lm-eval 0.4.11 `simple_evaluate` 的端到端运行（随机小模型 + 本地合成任务） | 生成与选择题请求跑通；SnapKV 与 FullKV 结果通过 `compare_results` 的输入一致性校验；结果、逐调用 KV 快照 CSV 与 `resolved_eval_<run_id>.yaml` 按运行编号写出 |

### 未验证

- 服务器上的 torch218 本身（真实 `PYTHONPATH`、editable litgpt 共存、conda 激活）。
- GPU 上的数值（CUDA SDPA、bf16 下 top-k 的并列取舍）、显存与速度；`majob.sh` 的 CUDA 自检与库路径在训练作业中的实际效果；多卡与多节点作业。
- 真实 Qwen3-1.7B-Base 权重、服务器上的离线 HF/RULER 缓存；`majob.sh --limit 4` 小样本、全量评测、与 SemanticLogKV 的真实样本对照（`compare_results`）均未执行。
