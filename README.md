# PromptSentinel

面向提示注入检测的轻量旁路模型实验。项目以 [Protect AI DeBERTa](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) 为二分类基座，在 [PromptShield](https://huggingface.co/datasets/hendzh/PromptShield)（ACM CODASPY 2025）官方 `train / validation / test` 划分上，对比 LoRA 与全量微调（Full FT）。模型只读取待检测文本，输出提示注入风险分数，不读取系统提示词、用户任务或模型回答。

## 实验摘要

- 数据：PromptShield 固定 revision 的官方划分，训练 / 验证 / 测试记录分别为 17,966 / 949 / 23,516 条。
- 对照：Base Guard、LoRA Guard、Full FT Guard 共用基座、tokenizer、最大长度（512）和评测脚本；LoRA 与 Full FT 均使用共同随机种子 42、2024、3407。
- 主指标：在冻结的官方 test 上计算 `Recall@1% sample-FPR`；ROC-AUC、PR-AUC 基于全部逐样本分数计算。
- 结论：LoRA 在三项主指标的三 seed 平均值上均高于 Full FT，且在独立能力保持测试的 seed 42 初步结果中遗忘更少。因此，当前推荐 LoRA 作为该旁路分类器的微调路径。

## PromptShield 主结果

| 模型 | ROC-AUC | PR-AUC | Recall@1% sample-FPR |
| --- | ---: | ---: | ---: |
| Base Guard | 0.7037 | 0.4397 | 1.77% |
| LoRA Guard seed 42 | 0.9384 | 0.8719 | 43.96% |
| LoRA Guard seed 2024 | **0.9450** | **0.8873** | **56.07%** |
| LoRA Guard seed 3407 | 0.9308 | 0.8636 | 51.39% |
| Full FT Guard seed 42 | 0.9370 | 0.8638 | 33.26% |
| Full FT Guard seed 2024 | 0.9381 | 0.8658 | 32.82% |
| Full FT Guard seed 3407 | 0.9225 | 0.8449 | 36.89% |

| 训练方式 | ROC-AUC（均值 +/- 样本标准差） | PR-AUC（均值 +/- 样本标准差） | Recall@1% sample-FPR（均值 +/- 样本标准差） |
| --- | ---: | ---: | ---: |
| LoRA Guard | **0.9381 +/- 0.0071** | **0.8743 +/- 0.0120** | **50.47% +/- 6.11%** |
| Full FT Guard | 0.9325 +/- 0.0087 | 0.8582 +/- 0.0115 | 34.33% +/- 2.24% |

LoRA 在主指标上比 Full FT 高 **16.14 个百分点**。全部主测试均在冻结的 PromptShield test 完成，test 不参与 checkpoint 选择、阈值选择、清洗规则或超参数搜索。

## 原始能力保持

为衡量微调是否破坏基座已学分类边界，使用基座模型卡列出的 [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) 独立数据集。合并官方 train、test 并按文本 SHA-256 去重后，评测集为 1,291 条。遗忘率定义为：Base 判断正确、微调模型判断错误的记录数，占 Base 判断正确记录数的比例。

| 模型 | ROC-AUC | PR-AUC | Balanced Accuracy@0.5 | F1@0.5 | 遗忘率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Guard | **0.9830** | **0.9826** | **0.9111** | **0.9032** | **0.00%** |
| LoRA Guard seed 42 | 0.9536 | 0.9525 | 0.7709 | 0.7094 | **16.17%** |
| Full FT Guard seed 42 | 0.9529 | 0.9481 | 0.6928 | 0.5637 | 24.68% |

该部分目前只完成 seed 42，因此“LoRA 遗忘少于 Full FT”是初步证据，不替代主实验的三 seed 结论。

## 边界

主结论只覆盖 PromptShield 的逐样本提示注入分类，不能直接推导为真实 RAG、工具 / MCP 返回或端到端 Agent 的防护效果。后续会将 RAG 外部文本和 Agent 场景作为独立专项，分别冻结数据转换、阈值和指标，避免与主结果混算。

## 开源模型

最优的 LoRA seed 2024 adapter 已发布至 Hugging Face：
[haominglan/PromptSentinel-DeBERTa-LoRA](https://huggingface.co/haominglan/PromptSentinel-DeBERTa-LoRA)。
该仓库只包含 adapter、分类头与 tokenizer，不重分发基座权重；完整加载方式、许可证和模型边界见其模型卡。

## 复现

环境要求：Python 3.12、[uv](https://docs.astral.sh/uv/)。训练与评估会下载公开的 Hugging Face 模型和数据集。

```bash
uv sync

# 下载 PromptShield 固定 revision，完成 split 泄漏检查和长度审计。
uv run python -m src.preprocess.main --config configs/data.yaml

# 训练两条对照路径。
uv run python -m src.train.main --config configs/lora.yaml
uv run python -m src.train.main --config configs/full_ft.yaml

# 在固定 PromptShield test 上评估候选。
uv run python -m src.evaluate.main --config configs/evaluate.yaml
```

运行产物统一写入 `outputs/`，包括配置快照、训练曲线、checkpoint、逐样本预测、评估指标和执行日志；该目录不会进入 Git。随机种子实验使用 `configs/lora_seed_*.yaml` 与 `configs/full_ft_seed_*.yaml`。

## 目录

```text
configs/       可提交的训练、数据与评估配置
data/          本地原始数据、prepared 数据与审计结果（不提交）
src/core/      跨模块配置、数据实体与训练枚举
src/preprocess/ 数据下载、校验与 prepared 数据集构建
src/train/     训练任务、策略对象、模型实现与训练可视化
src/evaluate/  PromptShield 与能力保持评估
outputs/       本地运行产物（不提交）
```

## 贡献与安全

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，社区行为规范见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。安全问题请遵循 [SECURITY.md](SECURITY.md)，不要在公开 Issue 中披露可复现攻击细节或敏感样本。

## 许可证

本仓库代码以 [Apache-2.0](LICENSE) 发布。训练数据、基座模型和 Hugging Face adapter 分别保留其模型卡、数据集卡和上游许可证要求；本仓库不包含数据集或模型权重。
