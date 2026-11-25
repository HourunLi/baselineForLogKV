# ChunkKV 与 niahLogKV 的评测对照

参考版本为 niahLogKV 0f4396b。配置来自两个模型的 `eval_full_baseline.yaml` 继承链，默认 B=128、recent=2048、K=12、exact=1024。ChunkKV 使用所选规模的原始 Qwen3 Base。niahLogKV 和它的 dense 同权重对照在 1.7B 和 8B 上分别使用 `qwen1.7b-32k-gamma_k12_b128_p1024_2k` 和 `qwen8b-32k-gamma_k12_b128_p1024_2k` 的 CPT 权重。优先使用目录自身的权重，缺少时按参考规则解析最新的 `step_*`。每份结果记录实际二进制路径及 SHA-256。比较 niahLogKV 与 CPT dense 时，两者的实际权重哈希应一致。比较 ChunkKV 与 niahLogKV 时，分数差异同时包含权重和缓存策略的影响。指定 `--model /path/to/step_N --tokenizer /path/to/Qwen3-X-Base` 可固定某一步和 tokenizer。

`--backend chunkkv|logkv|dense` 复用同一份 niahLogKV 请求与任务代码。dense 默认对应参考项目的 CPT dense 对照。若要测量 ChunkKV 在原始 Base 上单独引入的分数变化，可用 `--backend dense --model /path/to/Qwen3-X-Base`。各后端的模型和任务生成器默认共用原始 Base tokenizer。复现实验时三个后端采用相同的 `--model-size`、`--reference`、数据目录与样本限制。默认的 `--reference baseline` 使用 K/V 分数准入，也可选择 `word_first`、`word_first_query` 或 `word_first_query_dual` 对应的准入配置。ChunkKV 从所选配置的初始常驻缓存读取预算，自身用观察窗注意力选择块。

## 各档预算

旧流式实现在每一层持久保留了“预算 + 首块长度”的空间，后续增长还会增加备用容量。原 32K 单次压缩模式在生成时追加 KV。两者都会让仅按 prompt 预算计算的显存指标偏低。

完整对照现在每层预分配固定容量。设 `R` 为参考缓存按底层 storage 去重后的初始字节数，排除共享 RoPE 表。对 batch=1，设 `L` 为层数，`G` 为 KV 头数，`D` 为 head_dim，`b` 为元素字节数，则每个可保留位置的总成本为 `C = L × (2 × G × D × b + 8)`。末尾 8 字节是每层一个 int64 绝对位置。容量取 `floor(R/C)`，用实际 K/V 和索引 storage 字节数校验。待选的“旧条目 + 新块”只在当前层使用。每次请求重置有效长度并复用这些固定缓冲区。

下表用参考代码实际在 CPU 分配一层 BF16 缓存，再乘以 28 层或 36 层。两模型都用 8 个 KV 头和 128 维 head。表中百分比为 ChunkKV 常驻字节除以同窗口 dense K/V 字节，各档四舍五入到两位小数均与参考相同。正式运行会重新按加载的模型计算。

| 阶段 | 模型窗口 | 每层容量 | 1.7B 参考 / ChunkKV 字节 | 8B 参考 / ChunkKV 字节 | 占 dense KV |
|---|---:|---:|---:|---:|---:|
| 原生主线 | 32768 | 17682 | 2031878912 / 2031873984 | 2612415744 / 2612409408 | 54.07% |
| 32K | 33792 | 17682 | 2031878912 / 2031873984 | 2612415744 / 2612409408 | 52.43% |
| 64K | 66560 | 19286 | 2216205824 / 2216192832 | 2849407488 / 2849390784 | 29.03% |
| 128K | 132096 | 20916 | 2403505664 / 2403499392 | 3090221568 / 3090213504 | 15.86% |
| 256K | 263168 | 22571 | 2593778432 / 2593678752 | 3334857984 / 3334729824 | 8.59% |
| 512K | 525312 | 24253 | 2787024128 / 2786960736 | 3583316736 / 3583235232 | 4.63% |
| 1M | 1049600 | 25961 | 2983242752 / 2983230432 | 3835597824 / 3835581984 | 2.48% |

报告中的 `cache_budget` 给出常驻字节及相对 dense 比例，`kv_compression` 给出实际请求结束时的逻辑条目。整卡峰值还包含共同的权重、RoPE，以及因方法而异的注意力、选块、打包、输出 logits 和分配器工作空间。这部分须在相同设备、样本和软件栈上读取 `inference_metrics` 中的 allocated / reserved 峰值。当前 CPU 验证只支持上表的预算结论。按整块筛选还会留下少量空槽。流式 ChunkKV 可在 prefill 过程中淘汰条目，后续块能看到的历史与论文的单次完整 prefill 有差异；论文模式可用历史入口复现。

## 评测内容与输出

- 完整运行先跑原生 32K 主线的 49 个主入口及 20 个 NIAH 入口。主线 NIAH 按参考使用 1K–32K 六个长度。
- 长度阶段按 32K、64K、128K、256K、512K、1M 分开，窗口均预留 1024，默认 YaRN 倍数为 1、2、4、8、16、32，32K 关闭缩放。每任务最多 100 条，主线已覆盖的 32K 项按参考计划去重。
- 各档支持 RULER、BABILong QA1–5、LongBench v2、多轮 NIAH、单点长文 PPL 和书籍 PPL 曲线。RULER 从 256K 起移除 SQuAD，512K/1M 使用参考的扩展生成协议。InfiniteBench 到 256K。书籍曲线各点预测末尾 4096 tokens。LongBench v2 使用各档截断上限，实际输入可短于档位。
- 单独选 512K/1M 时，默认运行 RULER、BABILong、单点 PPL、书籍 PPL、多轮 NIAH 和 LongBench v2。显式指定的任务遵循上游必跑规则；完整默认清单中缺数据的可选任务会记录跳过原因，比较时核对实际结果中的任务和样本。
- 使用参考的离线数据与 tokenizer，按任务设置 `HF_HOME`（或 `CHUNKKV_HF_CACHE`）、`BABILONG_DIR`、`LONGBENCH2_DIR`、`INFINITEBENCH_DIR`、`LONG_PPL_DATA`、`LONG_PPL_BOOKS`、`LONG_PPL_DOCS` 和可选的 `RULER_DATASET_CACHE_DIR`。预生成 RULER 缓存沿用参考源文件指纹。参考数据协议
- 从仓库根执行 `python -m chunkkv.niah --help`。用 `--print-config` 在没有 GPU 和权重时审阅计划。每个阶段的输出目录独立，例如 `long1m/main` 和 `long1m/niah`。结果保留参考的任务分数、逐请求 KV 明细、首 token 延时、吞吐量与 CUDA 峰值。对同一阶段的两个结果可运行 `python -m chunkkv.compare_results A.json B.json` 核对实际输入与协议。
