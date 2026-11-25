# 512K / 1M 评测

全量评测默认包含 512K 和 1M。对两个模型目录中同名的 `eval_full_<method>.yaml` 设置
`EVAL_LENGTHS=512k,1m`，可以只跑 524288 和 1048576 tokens 两档。

```bash
EVAL_LENGTHS=512k,1m bash eval.sh exp/qwen1.7b-32k/eval_full_word_first_query.yaml
EVAL_LENGTHS=512k,1m bash eval.sh exp/qwen8b-32k/eval_full_word_first_query.yaml
```

单独选择 512K、1M 或同时选择两档时，默认跑 RULER、BABILong、books 和 LongBench v2，
每任务 100 条。可用 `EVAL_SUITES` 选择任务，用 `EVAL_MAIN=1` 加跑 32K 主线。
窗口额外留 1024 tokens，YaRN 以训练长度 32768 为原长，factor 为 16、32。
结果写到方法原目录下的 `long512k/` 和 `long1m/`。省略 `EVAL_LENGTHS` 时，
全量配置运行 32K 主线和 32K、64K、128K、256K、512K、1M 六档，并按默认任务清单安排逐轮问和单点 PPL。
InfiniteBench 的任务只注册到 256K。

## 数据准备

已有 512K/1M 评测数据时，在联网并且能访问评测共享缓存的节点只补下载 CWE 词表：

```bash
bash diag.sh unused/fetch_long_eval_data.py --lengths 512k,1m \
  --ruler-cwe --no-babilong --no-ruler-qa --out fetch_ruler_cwe.json
```

命令在 `${RULER_CACHE_DIR}/english_words.json` 保存校验过 SHA256 的普通文件，两个长度共用。
有可用文件时直接复用。FWE 在评测时本地生成编码词，只需已有 tokenizer。
全量数据准备也会自动下载这份词表。

下载与评测节点必须看到相同的普通文件目录。以下在仓库根目录执行，下载脚本默认绕过证书检查；
不要加 `--verify-ssl`。Hub 的下载 URL 固定到本次查询的 revision，逐文件核对大小和 SHA256；
新数据流程不使用 snapshot 软链接，也不使用目录重命名发布。
PPL 的 32K–256K 下载流程同样使用普通文件；可用的 Hub 缓存先核对大小、SHA256 和 Parquet 头尾再复制，
缺失或校验失败才重新下载。旧 `LONG_PPL_BOOKS` 若指向 `hub/datasets--.../snapshots/...`，
下载器改用普通目录并打印新路径；评测进程必须导出新路径，子进程无法修改父 shell 的旧环境变量。

```bash
export HF_HOME=/workspace/data/hf_cache
export LONGBENCH2_DIR="${HF_HOME}/longbench2"
export BABILONG_DIR="${HF_HOME}/babilong"
export LONG_PPL_DOCS=10
export LONG_PPL_BOOKS="${HF_HOME}/long_ppl_books_1m/emozilla--pg19"

bash diag.sh unused/fetch_long_eval_data.py --lengths 512k,1m --longbench2 \
  --ppl-books-hub emozilla/pg19 \
  --tokenizer /workspace/checkpoints/Qwen/Qwen3-8B-Base \
  --out fetch_512k_1m.json

bash diag.sh unused/check_long_eval_data.py --lengths 512k,1m \
  --suites ruler,babilong,books,longbench2 \
  --tokenizer /workspace/checkpoints/Qwen/Qwen3-8B-Base
```

1.7B 将 tokenizer 改为 `Qwen3-1.7B-Base`。每个 tokenizer 有独立的 PPL 选择缓存。
必须确认所选书籍未用于当前 CPT；上面的 PG19 train 只是下载接口示例，不保证与训练集无重叠。
有已分离的长文时，改用 `--ppl-books-data <普通 parquet/jsonl 文件或目录>`。
下载器逐 shard 扫描，直到够 `LONG_PPL_DOCS` 本或遍历完来源；不足时明确报告实际数量，
没有任何合格长文时报错。不同方法必须使用同一目录、tokenizer 和 `LONG_PPL_DOCS`。

下载日志实时打印并写入 `fetch_512k_1m.summary.txt`，可在另一终端用
`tail -f fetch_512k_1m.summary.txt` 查看。日志显示阶段开始/结束、文件大小、已下载字节和速度、
SHA256 校验、BABILong JSON → Arrow，以及 PG19 的 shard/文档分词和合格数量。
耗时调用期间每 20 秒打印 `WAIT`、已耗时间和最近状态；它表示进程仍在该阶段，
不能单凭 `WAIT` 判定远端网络在传输。读超时及重试仍按原设置执行，429 会单独报告等待时间。

该命令先准备 RULER，再准备 BABILong、LongBench v2，最后扫描 PG19。
512K/1M 的 BABILong 共 10 个大 JSON，还需写 Arrow；PG19 按文件名下载完整 shard，
对候选长文分词后才能判断是否达到 1048576 tokens，耗时可能明显超过单次下载。
务必保留上面的 `export LONG_PPL_DOCS=10`：不设置时目标是 100 本，可能扫描更多 shard。
这个值也必须带到评测环境。成功下载且校验通过的普通文件会复用；中断留下的不完整文件会重新下载。
旧版本只在结束时统一打印摘要，`python -u` 本身不能补出这些尚未打印的状态；
拉取更新后须重新启动下载进程，已有进程不会自动获得新日志功能。
读取 Parquet 前还会打印 `[PPL input]` 的实际路径、大小、前 128 字节和末尾 4 字节。
读取端只读解析已有路径占位文本，不修改共享 snapshot；断链、循环和坏 Parquet 会明确报出文件，
不会被跳过。离线检查也核对原文件，不会仅因命中文档选择缓存就通过。

PPL 分词后的文档选择缓存仍按原 key 复用，写入先使用系统临时目录中的独有普通文件，
再复制到 `LONG_PPL_CACHE`；不在共享 bucket 创建 PID 命名的临时文件，也不依赖 rename 或软链接。
共享目录暂时不可见时短暂重试，仍失败则警告并使用本次已选出的文档继续评测。
缓存不可读或 JSON 尚未复制完整时重新从原数据选择；原数据读取失败仍会报错。
希望完全绕过共享挂载缓存写入时，可在启动评测前设置（只改变分词结果缓存位置）：

```bash
export LONG_PPL_CACHE=/tmp/logkv_long_ppl
```

各节点会独立建立本地缓存；原始 `LONG_PPL_DATA` / `LONG_PPL_BOOKS` 路径保持各自设置。

可单独准备：

```bash
bash diag.sh unused/fetch_long_eval_data.py --lengths 512k,1m --no-ruler-qa
bash diag.sh unused/fetch_long_eval_data.py --ruler-essays --no-babilong --no-ruler-qa
```

前者下载 BABILong qa1–qa5 官方 `512k`、`1M` JSON，各 100 条，并生成普通 Arrow 流文件。
评测优先以内存映射读取 Arrow，不创建 datasets builder 缓存；只有手工 JSON 时可只读加载 JSON。
后者准备 `RULER_CACHE_DIR/paul_graham_essays.txt` 和普通 NLTK punkt_tab 文件。
LongBench v2 两档共用原有 20 子集、503 道题，不另下一个长度版本。

## 在 CPU 作业预生成 RULER 文本

准备好背景文章、HotpotQA、CWE 词表和 NLTK 文件后，在 CPU 作业并行构建 RULER 和逐轮问样本。
`TOKENIZER_DIR` 指向评测使用的 tokenizer 目录，预生成和评测使用相同仓库版本、Python 环境及数据缓存。

```bash
export RULER_DATASET_CACHE_DIR="${HF_HOME}/ruler_cache/generated"
bash diag.sh unused/prepare_ruler_datasets.py --tokenizer "$TOKENIZER_DIR" \
  --lengths 1m --samples 100 --workers 4 --cache-dir "$RULER_DATASET_CACHE_DIR"

EVAL_LENGTHS=1m EVAL_SUITES=ruler,turns \
  bash eval.sh exp/qwen1.7b-32k/eval_full_word_first_query.yaml
```

GPU 作业需读取同一个共享目录。缓存按生成代码、tokenizer、背景数据、长度和条数区分，
命中后各 rank 直接映射同一个 Arrow 文件。显式设置 `RULER_DATASET_CACHE_DIR` 时，缺少匹配文件会立即报错。
默认也会查找 `RULER_CACHE_DIR/generated`，缺失时各节点在本机临时目录只构建一份。
`LIMIT=2` 试跑优先读取已有全量缓存，首次构建时最多生成 2 条。全量缓存保留每个任务的原始条数。
缓存缩短样本准备阶段。GPU 推理阶段的首 token 时间和后续解码时间见评测的推理速度报告。

## 任务口径

- RULER：512K/1M 的捞针与逐轮问使用 `extended-niah-fit-v1`，变量追踪使用 `extended-vt-fit-v1`。
  先估算背景词数或条目数，再按实际 token 数调整，下一条样本复用上一条的背景条目数作为起点。
  保留上游模板、插针和变量链生成规则，并为每条样本设定可复现的随机种子。
  essay 背景不足时先循环背景词序列，再插针。默认每任务生成 100 条，可设 `RULER_LONG_SAMPLES`。
  旧 `extended-v1` 使用逐步累加探测，样本和新协议可能不同，横向比较时使用同一协议。
  CWE 的 `extended-cwe-nvidia-v1` 使用 NVIDIA RULER 的大词表，在现有模板下每个答案词出现 30 次，
  每个干扰词出现 3 次。FWE 的 `extended-fwe-fit-v1` 保留原词频分布，按实际 token 数细化词数搜索。
  HotpotQA 的 `extended-qa-fit-v1` 保留原问题、参考答案和题目关联的上下文，使用固定随机种子从原文档池
  无放回抽取干扰文档，按每条样本的实际 token 数调整文档数。扩展样本的长度包含答案前缀和生成预留。
  旧 `extended-cwe-v1` 属于等比例提高词频的历史协议。
  样本长度与协议字段见 [指标口径](evaluation-metrics.md)；不足目标长度 4096 tokens 以上时报错。
  保留已有 SQuAD 排除规则，因此超长档为 12 项，不能把均值直接当成其他论文完整 13 项的均值。
  32K–256K 的生成参数保持不变。此协议必须在论文中披露，不能称为原版 128K RULER 协议。
- BABILong：官方长度用其原始 tokenizer 定义，不保证 Qwen 编码后恰好达到相同 token 数。
  计分、两条示例和 16-token 生成上限保持原样。读取普通数据失败时报错，不回退到 Hub builder。
- 长文 PPL：512K/1M 都选择至少 1048576 tokens 的同一组文档。曲线点为 8K、16K、32K、64K、
  128K、256K、512K，以及 1M 档额外的 1M 点；每点预测最后 4096 tokens。
  相同点在两档使用相同文本，YaRN 设置不同。旧的 <=256K 曲线继续使用原来的 256K 文档集合。
  独立的 `long_ppl_512k`、`long_ppl_1m` 也可用 `EVAL_SUITES=ppl` 选择，来源为 `LONG_PPL_DATA`。
- LongBench v2：沿用全部原始问题，在各档窗口下评测。不能把所有短样本都称作实际 1M 输入。
- InfiniteBench：目前只注册到 256K，新长度默认不加入；显式要求时会报错，避免制造名为 1M 的短输入评测。

生成明细同时记录原始、截断后 prompt 长度和截去的 token 数。比较不同方法时核对实际输入、
样本数、权重路径和 `log_kv_dense_mode`。本次入口复用已有权重；8B 的完整 Dense KV 在 1M 时需要
远高于 32K 的显存，应按实际硬件安排，不能把 OOM 当作任务分数。
