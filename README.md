# Infini-Attention × Qwen3-8B-Base

本分支实现基于原始 Qwen3-8B-Base 的 GQA Infini-Attention。训练与 LogKV 8B 使用相同起点、32K ProLong 语言建模目标和 **8,388,608,000 个输入 token**。评测对齐固定版本的 niahLogKV，包括 32K 主线和 32K、64K、128K、256K、512K、1M 长测。8B 对照配置

## managed_cluster 一次提交训练和评测

选用上次 LogKV 的 CUDA / torch218 镜像与共享桶挂载，分配 **4 个节点，每节点 8 张 GPU**。使用本分支仓库根目录作为工作目录，每节点由 managed_cluster 执行一次同一条启动命令：

```bash
bash majob.sh --pipeline
```

作业先实际执行 32 个训练进程的 BF16 NCCL 求和，检查各节点的底模分片、tokenizer、ProLong 输入、共享输出和评测环境。主节点离线打开主线任务数据，并核查各档长测语料。已有兼容的 ProLong 缓存直接复用，只有缺缓存且原始语料可读时，主节点才在本次输出目录准备数据。所有节点完成 CPT，写出最终权重后，自动运行 32K 主线及 32K–1M 六档评测。任一阶段失败，全作业停止。单独使用 `bash majob.sh --check` 只执行预检，默认一键命令已包含它；预检可能生成数据集或分词缓存。

平台提供 `MA_NUM_HOSTS`、`VC_TASK_INDEX`、`MA_NUM_GPUS` 和主节点地址；沿用原作业的网络设置，节点间需能访问默认端口 6000–6002。每次输出根目录默认为 `/workspace/ckpt-logKV/nfini-attention/${MA_VJ_NAME}`（未设置作业名时使用 `qwen8b-32k`）。`cpt/` 为最终权重，`evaluate/` 为成绩。可用 `NFINI_OUTPUT_ROOT` 指定新的共享目录。训练已成功而仅需重跑评测时，设置 `NFINI_OUTPUT_ROOT` 为原作业根目录、`NFINI_OUTPUT_DIR` 为目标结果目录，然后运行原有的 `bash majob.sh`。

**下载与数据。** 启动器离线运行。共享桶已有原始 HF 格式的 Qwen3-8B-Base、ProLong 和完整评测资产，且镜像具备下述依赖时，无需下载模型、数据或安装 Infini-Attention 包。只有 LitGPT 的 `lit_model.pth` 时，需先补齐原始 HF 权重和 tokenizer。长测缺失的数据可在联网准备机按预检报告补齐，再挂载同一目录。

| 评测输入 | 沿用的 LogKV 默认路径 |
| --- | --- |
| HF_HOME | `/workspace/data/hf_cache` |
| 普通长文 PPL | `$HF_HOME/long_ppl_data/emozilla--pg19-test` |
| 32K–256K 书籍 PPL | `$HF_HOME/long_ppl_books/emozilla--pg19`，上限 100 本 |
| 512K/1M 书籍 PPL | `$HF_HOME/long_ppl_books_1m/emozilla--pg19`，上限 10 本 |

已导出的 `LONG_PPL_DATA`、`LONG_PPL_BOOKS`、`LONG_PPL_DOCS` 对所有档位生效。仅长两档使用了不同设置时，可以用 `NFINI_LONG_PPL_DATA_1M`、`NFINI_LONG_PPL_BOOKS_1M`、`NFINI_LONG_PPL_DOCS_1M` 指向上次对照的取值。普通 PPL 与书籍 PPL 各自都需要至少一篇经同一 tokenizer 分词后达到 1,048,576 token 的文档。上次只做了书籍 PPL 时，普通 PPL 的 1M 数据仍需实际核实；若无合格样本，预检会报出具体来源并停止。LogKV 长测数据说明

## 单独训练

共享底模目录需有原始 Hugging Face safetensors、config 和 tokenizer。优先复用 LogKV 已有的、tokenizer 指纹一致的 ProLong 32K LitData 缓存。默认路径见 [CPT 配置](configs/nfini_cpt.yaml)。训练解释器需提供 PyTorch 2.11、Transformers 4.51.3、LitData 0.2.61 及项目依赖。可以通过 `NFINI_TRAIN_PYTHON` 指定单独准备的环境。

```bash
# 查看配置和预算，无需底模或 GPU
bash majob.sh --train cpt --print-config

# 仅在缺少既有 packed 数据时运行一次，输出到本项目的独立缓存
bash majob.sh --train prepare

# managed_cluster 每节点一次。正式对照作业共 32 个进程
bash majob.sh --train cpt
```

默认从原始 HF 权重装载，各层只新增 32 个门控。全部 8,190,736,512 个参数参与训练。每条 32,769-token 数据块形成 32,768 个预测位置。全局 batch 128、每卡 batch 1、32 进程累积 4 次，共 2,000 步。使用 BF16、分层 FSDP、分段激活检查点和分块交叉熵。AdamW 的 betas 为 (0.9, 0.95)，eps 为 1e-8，衰减组为 0.1。前 100 步线性预热至 5e-6，此后按 LogKV 的零起始步号执行余弦计划，最低学习率为 3e-6。

可通过 `NFINI_QWEN_BASE`、`NFINI_PROLONG_DATA`、`NFINI_PROLONG_CACHE_ROOT`、`NFINI_OUTPUT_ROOT` 分别覆盖底模、原始数据、包含 `litdata_<指纹>_ctx32768` 的缓存父目录和共享输出根目录。需要多个训练节点看到相同路径。默认每 400 步保存模型分片、优化器和各 rank 的数据位置。恢复使用相同配方和 world size。最终导出目录为 `checkpoints/nfini-attention/cpt`；`training.json` 含底模和产物的哈希、缓存索引、训练步数、实际 token、训练耗时和各 rank 峰值显存。

```bash
# 另设目录完成一个真实优化器步，沿用正式全局 batch；结果导出到 trial-1
NFINI_OUTPUT_ROOT=checkpoints/nfini-smoke bash majob.sh --train cpt --steps 1

# 示例，显式从已完整写出的第 800 步续训
bash majob.sh --train cpt --resume checkpoints/nfini-attention/step-800
```

## 分段与预算

每段 2,048 token。在 Qwen 原有 Q/K 归一化和 RoPE 后，段内做因果 SDPA，所有查询只读取此前完整段的记忆。令 φ(x) = ELU(x) + 1。段末按 KV 组执行 M ← M + φ(K)ᵀ V，z ← z + Σφ(K)。同组 4 个查询头共享一份 M、z，各头独立计算 φ(Q)M / φ(Q)z，再与本段输出作门控混合。每头的记忆系数初始化为 sigmoid(−2)；可训练偏移量从零开始。CPT 梯度跨段传播，每个新训练样本或独立评测请求重新初始化记忆。

推理只持有当前段的精确 K/V 和每层 8 组 FP32 矩阵、归一化向量。batch 1 时满额局部缓存 **288 MiB**，记忆 **18.140625 MiB**，总持久分配 **306.140625 MiB**，六档相同。报告分别给出实际有效的局部 KV 槽位、状态字节、持久分配字节、CUDA allocated/reserved 峰值。Dense 分母按同一批实际进入模型的 token 和 8B 的 36 层、8 个 KV 头、128 维计算。

## 离线评测

评测在既有 torch218 环境运行，依赖 lm-eval 0.4.11 和 datasets 4.8.4。使用 `NFINI_CHECKPOINT_DIR` 选择训练产物，`NFINI_OUTPUT_DIR` 选择独立结果目录。可复用 LogKV 的 HF_HOME、LONG_PPL_DATA、LONG_PPL_BOOKS、BABILONG_DIR、LONGBENCH2_DIR 和 INFINITEBENCH_DIR。完整运行需要对应档位的全部原始资产；任务缺失会报告错误。[数据与环境](docs/BASELINE_ENVIRONMENT.md)

```bash
bash majob.sh --print-config
LIMIT=1 EVAL_LENGTHS=none bash majob.sh configs/nfini_eval.yaml boolq niah_single_1
# 每个节点一次，训练完成后运行
bash majob.sh
# 仅检查两档 RULER 长测
EVAL_MAIN=0 EVAL_LENGTHS=512k,1m EVAL_SUITES=ruler LIMIT=1 bash majob.sh
```

主线有 49 个 main 入口和 20 个 NIAH 入口，主线 NIAH 在 1K–32K 的六个长度运行。长测按固定计划对当次已执行的任务去重，每档窗口为目标长度加 1,024，默认每任务上限 100 条。RULER、BABILong、多轮 NIAH、长文 PPL、书籍 PPL 和 LongBench v2 覆盖六档。RULER 的 SQuAD 到 128K，InfiniteBench 到 256K。默认 YaRN 按目标长度相对 32K 的倍数设置，32K 保持原始 RoPE。评测只读指定 checkpoint。

计时与质量结果在各阶段的时间戳 JSON 中；`.summary.csv` 包含各任务和任务组的原始 metric、filter、stderr 与输入摘要，XLSX 包含速度与显存。质量比较使用相同任务、样本、prompt、tokenizer 和截断口径，另报双方的 GPU 小时。短任务可用同一批样本比较原始 dense Qwen、LogKV 和本方法的分数。1M 单针按实际保留 prompt 的 token 位置汇总前、中、后各三分之一及最前 10% 的正确数和样本数；被切掉的针单列。下例的输入是正式结果 JSON。

```bash
python -m nfini_attention.report /path/to/logkv_result.json /path/to/infini_result.json > comparison.csv
python -m nfini_attention.depth /path/to/1m_niah_result.json --tokenizer /path/to/qwen_tokenizer > depth.json
```

## 验证范围与来源

`python -m pytest -q tests/test_nfini_attention.py` 检查原始权重迁移、关闭记忆后的段内 dense 对齐、独立公式重算矩阵、因果性、跨段分块、缓存重置、CPT 保存恢复及 lm-eval 的三种请求。开发机另外用双进程 CPU FSDP 验证了连续与断点恢复训练逐参数一致，用小型 BF16 模型完成 32K 前向与反向。在微型模型上使用正式的 1M 长文本 PPL 入口、合成留出文本和 RULER 单针生成任务，验证了打分、生成和位置统计。8B 在目标 GPU 的完整训练、质量、峰值显存和吞吐仍待实测。

- 方法采用原论文的 Linear 更新和逐头门控。第三方 Qwen1.5-MoE 实现提供算法参考，本文的 Qwen3 分组记忆、请求缓存和训练代码在本项目内实现。[论文](https://arxiv.org/abs/2404.07143) · [参考实现及固定提交](https://github.com/jlamprou/Infini-Attention/tree/3a86dd5c528f3cf9ecd231314338c5fac6f88dd1)
- 长测计划、任务定义和评测协议源自 Apache-2.0 的 niahLogKV；项目内已有评测器和 ProLong 读取器在本分支复用。CSV 汇总沿用本仓库 5e00e96 的报告实现，按新模型提供缓存字段。运行时使用本仓库私有模块和已安装的依赖。固定评测版本 · [已有代码来源](lla_bench/SOURCES.md) · [许可证](LICENSE)
