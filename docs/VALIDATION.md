# 本地验证记录

日期：2026-10-12。分支：`DMS`。

环境：macOS，Python 3.11.0、PyTorch 2.9.1、Transformers 4.57.3、accelerate 1.4.0。首次训练组件检查使用 datasets4.4.2；评测对齐检查切换到 datasets3.6.0、lm-eval0.4.9、LitGPT editable 核心包。依赖装在临时隔离环境，未修改系统 Python。没有可用 CUDA，也没有安装 FlashAttention。

## 已执行

```bash
python -m pytest -q tests/test_qwen3_base.py tests/upstream/test_dms_utils.py tests/test_qwen3_cuda.py
```

首次结果：**15 passed, 1 skipped**，4.14 秒。CUDA 检查因无 CUDA 设备跳过。

覆盖内容：

- Base 纯文本模板与两份配置；1.7B 的完整结构在 meta device 上构造，确认 516,096 个适配器参数以及 embedding 绑定。
- 从本地随机小型原始 Qwen3 权重加载 DMS，确认权重保留、门控投影置零和底模冻结。
- 小模型 teacher 输出与原始 Qwen3 一致；eager CPU attention 前向、反向产生有限且非零的适配器梯度，底模没有梯度。
- BF16 导出、AutoModel 动态加载、适配器值与绑定权重恢复；导出不修改训练模型 dtype。
- 分布式导出的接口模拟：每个 rank 参与收集，只有主进程写文件。**这不是实际 FSDP 多卡验证。**
- 本地 JSONL 分词拼接、空上下文报错、恢复路径选择、压缩目标日程。
- prefill CR 统计包括近期窗口；无淘汰时为 1，迁入 paged cache 后也保持正确统计。
- 上游门控函数的形状、二值前向、温度和梯度检查。

此外完成 editable 安装，在仓库外运行训练与推理入口的 `--help`，Python 与 shell 语法检查、本地文档链接检查。环境预检按预期拒绝无 CUDA 的正式训练，没有静默切换成其他算法。

## 本次评测对齐检查

```bash
python -m pytest -q tests/test_qwen3_base.py tests/test_prepare_base.py tests/test_semanticlogkv_eval.py tests/test_comparison_contract.py tests/test_readonly_adapter.py tests/upstream/test_dms_utils.py tests/test_qwen3_cuda.py
```

撤回参考仓库接点、改为 baseline 自身兼容接口后的结果：**39 passed, 1 skipped**，5.80秒；包括前述原有检查。新增验证覆盖：

- 用真实 tiny LitGPT 模型执行官方 Qwen3 映射，逐张量零误差比较，HF/LitGPT logits 一致；tied head 或结构错误拒绝。
- Base 来源、转换权重、模型配置、tokenizer 哈希验证；篡改拒绝，禁止覆盖输出；来源信息经过 DMS 导出/重载保留。
- 从转换后的 tiny Base 经官方 student 加载、保存，再用共享评测 DMS loader 严格重载；仅跳过 CUDA 环境检查，不执行 GPU 前向。
- 从当前 `litgpt/eval.py` 抽取真实请求方法，调用真实 LitGPT generate，仅用 fake HF 计算代替 GPU attention；检查绝对位置、logits 切片、截断、BOS、EOS、当前 until 行为、rolling 窗口和最后未回填 token 的计数。
- DMS prefill/decode 两区 KV 计数与共享比率；两后端配置相同；同名任务但 prompt/生成参数/环境不同，结果核对拒绝。
- 仓库边界：私有加载/接入 DMS 前后，参考文件字节一致，原生 `litgpt.kv_compression` 函数身份不变；另一次加载仍是原生 LogKV 类，请求方法保持直接继承。
- 输出隔离：原入口的临时 JSON 只写入 baseline 输出目录，禁止输出到参考仓库；缺少本次新结果时拒绝给旧结果补来源信息。

实际运行 DMS 和 LogKV 两种后端 `--print-config` 均成功；确认库实际使用的 HF cache 与共享环境设置相同，解析任务为 single2/3、窗口32768。完成依赖一致性检查（pip check）、Python 编译和 shell 语法检查。

此前对共享 LitGPT 运行压缩统计、请求处理、HF缓存别名、缓存诊断、通信测试：最初 **22 passed、2 skipped、2 failed**。其中 Gloo 失败来自沙盒禁止回环端口，允许本机通信后单独重跑 **1 passed**。剩余挂载诊断测试依赖 Linux `/proc/self/mountinfo`，在 macOS 报 `KeyError: mount`；已逐函数 AST 对比确认其实现与原 HEAD 完全一致。此处保留历史检查结果，不将整组选定测试表述为全通过。

边界修正完成后，`git diff --exit-code -- eval.py litgpt/kv_compression.py docs/evaluation-metrics.md` 在参考仓库返回0；这三处新增内容已撤回。其余原本存在的 README/tutorials 修改保留，没有做全仓库重置。

## 尚未验证

- 没有下载或加载完整 Qwen3-1.7B-Base 权重，没有得到训练完成的适配器。
- 未执行 CUDA/Triton/FlexAttention 编译、FlashAttention decode、真实单卡训练或 FSDP 多卡训练。
- 未测显存、吞吐、任务精度，未运行 RULER。未完整执行上游所有测试。
- 未验证真实服务器 Base 的指纹和缓存样本内容；代码会在实际运行时检查来源并保存逐样本记录。CPU facade 测试不等于真实 DMS CUDA 推理验证。

下一步在目标 CUDA 机器准备同一份 Base，检查 CUDA 内核并完成两步训练/导出链路；已有正式适配器时，优先通过统一入口运行双方各4样本，再用 `compare_results` 核对。合成试跑模型不用于报告精度。
