# PromptSentinel

提示注入旁路检测器微调工程。在 PromptShield 官方 `train / validation / test` 划分上，比较 Prompt Guard 2 的 LoRA 与全量微调。当前仓库只提供数据准备、训练和模型保存；测试集推理与指标报告后续放在 `src/test/`。

## 目录

```text
configs/       可提交的实验配置
data/          本地原始数据、训练 manifest 和长度审计报告
runs/          每次训练的模型、日志和复现快照（不提交）
src/core/       配置 dataclass、数据实体和训练枚举
src/preprocess/ PromptShield 原始数据与 prepared 数据集的对象编排
src/train/      Prompt Guard 2、训练策略、运行产物和训练任务对象
src/test/       后续测试集推理与指标任务对象
src/train/     数据准备与训练入口
src/test/      后续测试集推理和指标代码边界
```

## 入口

```bash
uv sync
uv run python -m src.preprocess.main --config configs/data.yaml
uv run python -m src.train.main --config configs/lora.yaml
uv run python -m src.train.main --config configs/full_ft.yaml
```

`prepare_data` 会下载 PromptShield 的固定 revision，检查 split 泄漏，并按 Prompt Guard 2 tokenizer 生成长度审计。若 `train` 或 `validation` 的超长样本占比高于配置门槛，命令会停止，不会静默修改训练语义。

Meta Prompt Guard 2 是 gated 模型。首次实际运行前，需在 Hugging Face 接受模型许可并完成本机登录。
