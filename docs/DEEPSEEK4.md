# 36 层 V4-Flash 注意力基线

## 模型

以原始 Qwen3-8B-Base 的 LitGPT 格式权重和 tokenizer 为起点。保留 4096 维残差接口、独立 embedding 与 LM head、36 层的 dense SwiGLU 和块级 RMSNorm。36 个注意力模块重新初始化。在 10,929,049,984 个参数中，6,680,776,704 个继承并冻结，4,248,273,280 个参与适配。训练后保存完整 BF16 模型。

层布局取 V4-Flash 主干压缩率列表的前 36 项，即 `[0,0]+[4,128]*17`。每个注意力层使用 64 个 512 维查询头、一个 512 维共享 K=V、64 维部分 RoPE、1024 维查询低秩投影、8 组输出投影及可学习的 attention sink。局部窗口 128。CSA 的逐通道门控压缩重叠两个 4-token 块，独立的 64 头 × 128 维 indexer 读取历史并选择最多 512 项。HCA 压缩非重叠的 128-token 块并访问所有已完成的块。局部条目与压缩条目共享 softmax。压缩项的位置取块首，只有看到块末才可访问。[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/2b2bebc1776b2cf88ca36f0b5b6f816ae0f5892f/config.json) · [官方注意力实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/2b2bebc1776b2cf88ca36f0b5b6f816ae0f5892f/inference/model.py)

实现使用 PyTorch。BF16 版本省去了官方量化模拟与配套的 Hadamard 变换。完全同分的 indexer 候选优先选较早的位置。纯滑窗层使用 theta 10000，压缩层使用 theta 160000，并将 YaRN 原始长度调整为 Qwen/niahLogKV 的 32768。Qwen 主体保留 dense MLP 和常规残差。发布结果时应注明迁移架构、精度、参数量和训练预算。

## 数据和训练

共享路径的默认值见 [8B 配置](../configs/deepseek4/qwen8b.yaml)。在仓库根运行。覆盖路径可设置 `DEEPSEEK4_BASE_CHECKPOINT_DIR`、`DEEPSEEK4_DATA_DIR`、`DEEPSEEK4_TOKENIZED_DIR` 和 `DEEPSEEK4_OUTPUT_DIR`。训练/评测作业可分别用 `DEEPSEEK4_PYTHON` 选择解释器。底模目录应含原始 `lit_model.pth`、`model_config.yaml` 和 tokenizer。小规模本地检查使用 `python -m deepseek4.evaluate --print-config`。

1. `bash majob.sh prepare` 在共享目录准备 ProLong 语料。复用时验证 tokenizer、训练长度、原始文件清单与缓存索引。多节点只由全局 rank 0 执行准备。
2. `bash majob.sh train` 在独立的新输出目录创建训练结果。多卡采用逐块 FSDP、FP32 主权重、BF16 混合精度和激活重计算。加载冻结的原始 Qwen 教师，学生采用 next-token CE、教师 logits KL 与 indexer 辅助交叉熵。Indexer 用所选压缩条目的注意力质量构造监督，在全部可见候选上归一化。每步用一个各 rank 独立分片的 microbatch，实际消耗记录于 `steps.jsonl`。
3. 默认 4096 长度、160 步、每卡每步 1 条，8 卡共 5,242,880 个 token，是首次检查适配是否有效的试训预算。不能据此预判 4.25B 新参数能恢复多少质量。可在独立配置里更改训练长度、步数、语料和输出路径。改变训练长度时使用独立的预处理目录。
4. 完成后写入 `lit_model.pth`、`model_config.yaml` 和 `training.json`。清单记录 Base 文件指纹、语料、版本、模型布局、训练步数、token 和训练 GPU 小时。训练中断可能留下部分文件；正式评测要求有完成清单和匹配的模型 SHA-256。

多节点需要共享资产和输出目录，使用 `MA_NUM_HOSTS`、`VC_TASK_INDEX`、`MASTER_ADDR`、`MASTER_PORT` 与 `MA_NUM_GPUS`。也支持 `NNODES`、`NODE_RANK` 与 `NPROC_PER_NODE`。训练与评测分开启动。评测使用现有 torch218 环境，检查 lm-eval 0.4.11 和 datasets 4.8.4。

## 评测

`bash majob.sh eval` 加载训练清单指定的模型，按顺序运行 49 个 main 入口、20 个 NIAH 入口及六个长度的去重任务。源协议为 niahLogKV@0f4396b。支持 RULER、BABILong、LongBench v2、多轮 NIAH、单点 PPL、书籍 PPL 曲线以及至 256K 的 InfiniteBench。RULER SQuAD 至 128K。多轮 NIAH 在同一次会话共享缓存。各长档窗口为目标长度 + 1024，默认逐档 YaRN 和每任务最多 100 条。完整运行和单选超长档的默认套件与参考入口一致。默认可选数据缺失时参考评测器输出原因；显式指定相应套件时应准备其数据。

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

本地验证覆盖 36 层 meta 模型精确参数量、小模型因果性、整段/分块/单 token 等价、重置、indexer 和压缩器梯度、真实 LitGPT 加载与教师蒸馏、训练权重回读和参考评测器请求路径，以及六档任务计划。未在目标 CUDA 平台训练 8B 主体或执行 32K–1M 真实任务。

评测快照的来源、许可证、包名和模型接入修改记录见 [SOURCE.json](../deepseek4/_vendor/niahlogkv/SOURCE.json)。数据准备与文件指纹代码复用本仓库 `9af3958` 的 Hybrid CSA，遵循根目录 Apache-2.0 许可证，迁入私有命名空间，并对不完整的预处理目录报错。
