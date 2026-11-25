# 36 层 V4-Flash 注意力基线

## 模型

以原始 Qwen3-8B-Base 的 LitGPT 格式权重和 tokenizer 为起点。保留 4096 维残差接口、独立 embedding 与 LM head、36 层的 dense SwiGLU 和块级 RMSNorm。36 个注意力模块重新初始化。在 10,929,049,984 个参数中，6,680,776,704 个继承自 Qwen，4,248,273,280 个重新初始化，全体参数参加 CPT。训练后保存完整 BF16 模型。

层布局取 V4-Flash 主干压缩率列表的前 36 项，即 `[0,0]+[4,128]*17`。每个注意力层使用 64 个 512 维查询头、一个 512 维共享 K=V、64 维部分 RoPE、1024 维查询低秩投影、8 组输出投影及可学习的 attention sink。局部窗口 128。CSA 的逐通道门控压缩重叠两个 4-token 块，独立的 64 头 × 128 维 indexer 读取历史并选择最多 512 项。HCA 压缩非重叠的 128-token 块并访问所有已完成的块。局部条目与压缩条目共享 softmax。压缩项的位置取块首，只有看到块末才可访问。[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/2b2bebc1776b2cf88ca36f0b5b6f816ae0f5892f/config.json) · [官方注意力实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/2b2bebc1776b2cf88ca36f0b5b6f816ae0f5892f/inference/model.py)

实现使用 PyTorch。BF16 版本省去了官方量化模拟与配套的 Hadamard 变换。完全同分的 indexer 候选优先选较早的位置。纯滑窗层使用 theta 10000，压缩层使用 theta 160000，并将 YaRN 原始长度调整为 Qwen/niahLogKV 的 32768。Qwen 主体保留 dense MLP 和常规残差。发布结果时应注明迁移架构、精度、参数量和训练预算。

## 数据和训练

共享路径的默认值见 [8B 配置](../configs/deepseek4/qwen8b.yaml)。在仓库根运行。覆盖路径可设置 `DEEPSEEK4_BASE_CHECKPOINT_DIR`、`DEEPSEEK4_DATA_DIR`、`DEEPSEEK4_TOKENIZED_DIR` 和 `DEEPSEEK4_OUTPUT_DIR`。训练/评测作业可分别用 `DEEPSEEK4_PYTHON` 选择解释器。底模目录应含原始 `lit_model.pth`、`model_config.yaml` 和 tokenizer。小规模本地检查使用 `python -m deepseek4.evaluate --print-config`。

完整作业从每个节点运行同一条命令：

```bash
bash majob.sh all
```

`all` 先在每节点用 torch218 检查 CUDA BF16、NCCL、依赖、Base/tokenizer、训练语料和离线评测资产。它通过 lm-eval 加载原生主线任务，检查 RULER 的背景与问答文件、BABILong 的各长度/分组、LongBench v2 的各子集和 InfiniteBench 的文件与首条样本。主节点按 Qwen tokenizer 验证 PPL 文档可达到对应长度，复用参考评测器的文档选择缓存。检查成功后读取对照作业的分片并训练。所有训练进程等待 rank 0 写完权重及指纹，评测进程验证指纹后进入主线和六档评测。任何阶段失败都会以非零状态退出。单独运行 `bash majob.sh check` 可在目标环境执行相同的训练前检查。

8B 参考 YAML 直接读取 `LONG_PPL_DATA` 和 `LONG_PPL_BOOKS`。请保留对照作业导出的两项实际路径，并沿用同一份 `LONG_PPL_DOCS` 和 `LONG_PPL_CONCAT` 设置。单点 PPL 和书籍 PPL 在 512K/1M 档都需要能提供 1,048,576 tokens 的来源。完整模式若缺少这些变量或读不到足够长的文档，会在训练前报错。InfiniteBench 沿用显式目录、普通文件和 Hub 缓存的参考查找顺序。8B 长档参考配置

1. 正式作业直接读取 8B LogKV 的 ProLong LitData 资产。按参照实现对 Qwen tokenizer 文件计算哈希，在 `DEEPSEEK4_DATA_DIR` 下定位 `litdata_<hash>_ctx32768`。`prepare` 核验索引和分片大小。两边使用同一底模、分片、LitData 版本、进程与 worker 数才能复现相同的输入顺序。每卡 16 个 worker，shuffle seed 42，块长 32,769，训练输入与目标相差一个 token。没有合成检索样本。
2. 全模型 CPT 使用 next-token CE，加 CSA 索引器从本层注意力质量计算的辅助交叉熵。没有外部 teacher。模型采用 BF16、FULL_SHARD、激活重计算，每个 micro-batch 同步分片梯度。词表交叉熵复用参考实现，按 1,024 个 token 分块。AdamW 的 betas 为 (0.9, 0.95)，eps 为 1e-8，weight decay 为 0.1，名称含 bias、norm 或 ln_ 的参数免衰减。前 100 次更新线性预热至 5e-6，之后按参考代码用已完成更新数计算余弦衰减，目标下限为 3e-6。沿用对照的无梯度裁剪设置。
3. 32,768 上下文 × 全局 batch 128 × 2,000 次优化器更新，共 8,388,608,000 个目标 token。8 卡、每卡 micro-batch 1 时累积 16 次。数据按相同配置最多遍历 10 个 epoch；如无法完成预算，作业报错。采用第 2,000 步权重作为正式评测点。8B 对照配置 · 优化器与更新循环
4. 每 400 步原子发布一个断点，包含各 rank 的权重分片、AdamW 状态和 LitData 读取位置。中断后用同一命令、代码、环境、卡数和输出目录续训。默认输出为 `deepseek4/cpt-32k-b128-2k`。五个断点加最终 BF16 模型合计约 350 GB，使用独立共享存储。每次尝试保存独立的 `steps-from-*.jsonl`。最终 `training.json` 记录底模、分片索引指纹、依赖、实际步数、token 和 GPU 小时；GPU 小时包含之前尝试已记入日志的训练时间。评测前验证最终权重指纹。
5. 两种方法的输入和更新预算相同。CSA/HCA 模型的参数量、初始化、索引辅助损失和实际算力成本属于方法差异。历史 LogKV 作业实际完成的 token 和 GPU 小时需从该次作业日志核对。64K–1M 的五个档位属于训练长度外推。

多节点需要共享资产和输出目录，使用 `MA_NUM_HOSTS`、`VC_TASK_INDEX`、`MASTER_ADDR`、`MASTER_PORT` 与 `MA_NUM_GPUS`。也支持 `NNODES`、`NODE_RANK` 与 `NPROC_PER_NODE`。一体化入口使用独立的 torchrun 进程依次启动各阶段，managed_cluster 环境可由作业名、任务名及主节点索引推导主机地址。单节点使用 standalone，多节点各阶段使用不同 rendezvous 端口。评测使用现有 torch218 环境，检查 lm-eval 0.4.11 和 datasets 4.8.4。

## 评测

`bash majob.sh eval` 加载训练清单指定的模型，按顺序运行 49 个 main 入口、20 个 NIAH 入口及六个长度的去重任务。源协议为 niahLogKV@0f4396b。支持 RULER、BABILong、LongBench v2、多轮 NIAH、单点 PPL、书籍 PPL 曲线以及至 256K 的 InfiniteBench。RULER SQuAD 至 128K。多轮 NIAH 在同一次会话共享缓存。各长档窗口为目标长度 + 1024，默认逐档 YaRN 和每任务最多 100 条。完整运行和单选超长档的默认套件与参考入口一致。单独使用 `eval` 时保留参考入口的可选任务规则。一体化入口调用 `eval --full`，此时所有受支持的套件均必跑。

```bash
bash majob.sh eval --print-config
bash majob.sh eval --checkpoint /path/to/trained --output /path/to/new-eval
bash majob.sh eval --no-main --lengths 512k,1m --suites ruler,turns,ppl,books,babilong,longbench2
```

可用 `HF_HOME`、`BABILONG_DIR`、`LONGBENCH2_DIR`、`LONG_PPL_DATA`、`LONG_PPL_BOOKS`、`LONG_PPL_DOCS`、`INFINITEBENCH_DIR` 指定与参照相同的离线数据。每阶段保留 `protocol.json` 及参考评测结果文件，包含实际任务、环境、权重、样本和推理指标。重新运行时用新的输出路径。Qwen 的原生上下文是 32K，六档评测使用同一训练产物。

## 缓存和验证

按一个共享 KV 头的逻辑存储位置计数。每层有最近的最多 128 个位置，每个 CSA 层再存 floor(T/4) 项、每个 HCA 层再存 floor(T/128) 项。逐次统计检查实际缓存游标与已回填 token 数一致，汇总使用总 dense 槽位除以总有效槽位。常驻字节另计 index key、局部 KV、压缩池和用于跨块压缩的尾部 hidden 状态。与 Qwen BF16 dense 字节比较时使用其原生 8 个 KV 头与独立 K/V。GPU 峰值包含权重、激活和临时工作区，由正式运行单列。

下表用实际缓冲区形状在 meta device 上统计，batch 为 1，有效输入为档位长度，分配容量含 1024 个额外位置。

| 长度 | 主 KV 槽位倍数 | 常驻缓存字节 | Qwen dense 字节 / 常驻字节 |
|---|---:|---:|---:|
| 32K | 7.959× | 212,082,688 | 23.495× |
| 64K | 8.084× | 394,797,056 | 24.860× |
| 128K | 8.149× | 760,225,792 | 25.622× |
| 256K | 8.181× | 1,491,083,264 | 26.025× |
| 512K | 8.197× | 2,952,798,208 | 26.233× |
| 1M | 8.206× | 5,876,228,096 | 26.338× |

`python -m deepseek4.budget` 可复算。默认 prefill 每次 256 tokens，可用 `DEEPSEEK4_PREFILL_CHUNK` 调整。Indexer 按 query 和历史块分块计算，保持全历史检索。长上下文仍需大量计算，本实现使用 PyTorch 参考算子，尚无目标 GPU 吞吐结果。

本地验证覆盖 36 层 meta 模型精确参数量、小模型因果性、整段/分块/单 token 等价、重置、indexer 和压缩器梯度、真实 LitGPT 加载与无教师全模型 CPT、训练权重回读和参考评测器请求路径，以及六档任务计划。另在 torch 2.11.0 的 CPU/Gloo 双进程小模型上完成FSDP 梯度累积、断点恢复后与连续训练逐参数一致、逐层 BF16 权重导出及严格回读。开发机运行 Python 3.11；平台记录为 Python 3.10.20、torch 2.11.0+cu128，两侧的 pyarrow 与 jsonargparse 小环境版本也不同。开发机测试环境缺少 LongBench v1 的 fuzzywuzzy 与 rouge，目标环境将使用自己的包在训练前检查。未在目标 CUDA 平台训练 8B 主体或执行 32K–1M 真实任务。

评测快照的来源、许可证、包名和模型接入修改记录见 [SOURCE.json](../deepseek4/_vendor/niahlogkv/SOURCE.json)。数据准备与文件指纹代码复用本仓库 `9af3958` 的 Hybrid CSA，遵循根目录 Apache-2.0 许可证，迁入私有命名空间，并对不完整的预处理目录报错。
