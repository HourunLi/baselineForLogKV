# CaM 与 niahLogKV 的 32K–1M 对照

CaM、dense 和 LogKV 默认读取同一个 Qwen3-1.7B Gamma CPT，使用原始 Base tokenizer、BF16、batch size 1。权重严格加载，各阶段记录实际 checkpoint、权重与 tokenizer 哈希。8B 用 `--model-size 8b`。更换 checkpoint 时三个后端传相同的 `--model`、`--tokenizer`。生成评测样本的 tokenizer 路径沿用参考 eval.sh 的 SAVE_DIR；若 tokenizer 资产只存放在 Base 目录，三方统一指定 `--metadata-tokenizer DIR`。离线数据由参考评测器加载。

## 运行与指标

在仓库根目录和原评测环境 torch218 运行。平台每节点执行一次。仅选择六档 RULER 时，`--no-main` 让 32K 的全部 13 个任务也在同一窗口及输出目录评测。完整协议按参考计划去重，主线已完成的任务不会重复计数。

```bash
bash majob.sh --no-main --suites ruler --print-config
bash majob.sh --no-main --suites ruler --lengths 32k --limit 2 --output outputs/smoke
bash majob.sh --backend cam --budget logkv
bash majob.sh --backend cam --no-main --suites ruler --budget target-bytes
bash majob.sh --backend logkv --no-main --suites ruler
bash majob.sh --backend dense --no-main --suites ruler
```

长文本窗口为名义长度加 1024。默认按档使用 YaRN，倍率为 1、2、4、8、16、32。32K 至 128K 有 13 个 RULER 任务，256K 起有 12 个，SQuAD 的材料无法填满更长窗口。512K 与 1M 使用上游扩展生成协议。完整长评测还含 BABILong、LongBench v2、多轮 NIAH、长文档及书籍困惑度，以及 256K 以内的 InfiniteBench。长评测默认每任务 100 条，数据必选和可选规则、评分与逐样本聚合沿用上游。仅选 512K/1M 的默认子集和其他按需选择由 `--suites` 控制。dense 的可运行长度取决于显存，所有 rank 都各自加载完整模型。见 参考长文本规划器。

结果 JSON 中，`kv_compression` 按实际处理 token 数量和实际保留槽总数汇总。`cache_budget` 记录整模型持久缓存字节、参考字节、名义长度分母及含 1024 余量的 dense 分母。`inference_metrics` 含按请求测量的 CUDA allocated / reserved 峰值、TTFT 与解码速度。模型权重、RoPE 和单层临时张量会进入运行峰值。

## 六档预算

下表使用 Qwen3-1.7B 的 28 层、8 个 KV 头、128 维和 BF16。参考值取自实际 CPU 缓存构造器，分配一个代表层并乘层数，排除共享 RoPE，参数取本项目的 eval_full_baseline 快照。正式运行会在设备上按当前模型实测参考缓存，再为 CaM 固定预分配。每个时间位置跨所有层及头，dense 需 114688 字节，CaM 的 FP32 累计分数使其增至 115584 字节。

| 名义长度 | 参考字节 | 等字节 CaM 槽数 | dense 名义 KV / 参考字节 | 目标倍率 | 目标字节 CaM 槽数 | 目标逻辑 CaM 槽数 |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 2031878912 | 17579 | 1.85× | 4.08× | 7969 | 8031 |
| 64K | 2216205824 | 19173 | 3.39× | 7.11× | 9145 | 9217 |
| 128K | 2403505664 | 20794 | 6.25× | 12.37× | 10513 | 10595 |
| 256K | 2593778432 | 22440 | 11.59× | 21.57× | 12058 | 12153 |
| 512K | 2787024128 | 24112 | 21.58× | 38.38× | 13554 | 13660 |
| 1M | 2983242752 | 25810 | 40.31× | 69.43× | 14985 | 15102 |

`--budget logkv` 用参考分配字节除以每个时间位置的 CaM 字节数，向下取整。LogKV 提前分配近期缓存的备用空间、层级容量、元数据和可复用的打包工作区，因此同一运行的逻辑槽位倍率可以明显高于分配字节倍率。含生成余量时，参考分配字节倍率按六档分别为 1.91、3.44、6.30、11.64、21.62、40.35。

`--budget target-bytes` 将名义 dense KV 字节除以目标倍率后折算成槽数。`--budget target-slots` 直接将名义 token 数除以目标倍率后取整。前者在 1M 分配约 1.61 GiB，参考预算约 2.78 GiB。后者适合对比上游的逻辑槽位指标。真实 RULER 长度包含分词和生成差异，需从逐请求记录计算实际压缩率。三种配置各自输出到独立的默认目录。

## 实现核对

官方流式 CaM 在当步注意力输出后淘汰最旧的非 sink token。合并概率由被淘汰项的累计注意力除以后继 16 项的平均累计注意力得到，裁剪到 [0,1]。被选中时把 V 均分给这 16 项；保留的 K 按缓存槽重新施加 RoPE。默认 `--merge-interval 1` 对应此更新时序。Qwen3 GQA 对共享同一 KV 头的 query 头概率取平均，累计分数使用 FP32。启动器固定 32 个 sink，随机种子按层隔离。较大的 `--merge-interval` 将淘汰推迟到块末，会改变注意力历史，单独报告。如需测量这种吞吐折中，可显式选 `--merge-interval 128 --prefill-chunk 256`。分块求解顺序合并的 V 递推，临时线性系统每边最多 64 个被淘汰项。见 [官方流式缓存实现](https://github.com/zyxxmu/cam/blob/fc80e64f460b343be96c87596f2494fc9d319c45/generate_task/streaming_llm/kv_cache_cam.py)。

独立检查 chunkKV 分支发现两处比较时需要处理的差异。其默认 ChunkKV 运行选 Base，dense 与 LogKV 选 Gamma CPT，权重哈希会不同。其流式 `streaming_keep` 按得分排列后用累计块长取前缀，当较高分块放不下时，其后的较短块即使装得下也被排除。CPU 复现中，预算 6、观察窗口 1、chunk size 4，较旧候选块长度依次为 4、4、1，分数依次为 12、8、1，当前代码保留位置 [0,1,2,3,9]，位置 8 可装入剩余槽。这会影响逻辑保留率；固定缓冲区仍分配 6 个位置。参见 ChunkKV 权重选择 和 候选块选择。

已验证 CPU 缓存字节、六档计划、小模型 YaRN 下无淘汰区间与 dense 的 logits、跨块合并，以及参考评测器的 checkpoint 加载、生成和对数似然接口。真实权重的六档任务分数、GPU 峰值和吞吐需在目标设备与数据上测量。
