# PromptSentinel

提示注入旁路检测器微调工程。在 PromptShield 官方 `train / validation / test` 划分上，比较序列分类模型的 LoRA 与全量微调。当前默认基座为 Protect AI 的开放 `deberta-v3-base-prompt-injection-v2`。

## 目录

```text
configs/       可提交的实验配置
data/          本地原始数据、训练 manifest 和长度审计报告
runs/          每次训练的模型、日志和复现快照（不提交）
src/core/       配置 dataclass、数据实体和训练枚举
src/preprocess/ PromptShield 原始数据与 prepared 数据集的对象编排
src/train/      模型训练器抽象、具体模型加载、训练策略、运行产物和训练任务对象
src/evaluate/   后续评估推理与指标任务对象
src/train/report/训练完成后的运行产物报告与独立重渲染入口
```

## 入口

```bash
uv sync
uv run python -m src.preprocess.main --config configs/data.yaml
uv run python -m src.train.main --config configs/lora.yaml
uv run python -m src.train.main --config configs/full_ft.yaml
uv run python -m src.evaluate.main --config configs/evaluate.yaml
uv run python -m src.train.report.main --run-dir runs/<训练运行目录>
```

`prepare_data` 会下载 PromptShield 的固定 revision，检查 split 泄漏，并按当前配置的模型 tokenizer 生成长度审计。若 `train` 或 `validation` 的超长样本占比高于配置门槛，命令会停止，不会静默修改训练语义。

训练和测试的终端输出、第三方库日志和异常堆栈会与各自的配置快照和结果一起写入 `runs/<运行标识>/`。预处理日志写入 `logs/preprocess/execution.log`。

训练成功后，`TrainingJob` 会自动读取刚落盘的 `trainer_state.json`，在对应运行目录写入 `plots/loss_curve.png` 和 `plots/loss_curve.json`；报告不修改模型、checkpoint 或训练状态。`src.train.report.main` 仅用于为历史 run 重建图表。

评估统一在 validation 的良性样本上校准目标 FPR 阈值，随后固定该阈值在独立官方 test split 上比较所有候选。每次评估保存 `metrics.json`（AUC-ROC、AP、FPR、Recall 与混淆矩阵）和每个候选的逐样本 `predictions/*.jsonl`。`configs/evaluate.yaml` 默认比较基座和当前 LoRA adapter；加入完整微调结果时只需增加一个 `full_fine_tuned` 候选。

当前默认的 Protect AI 基座为开放模型。Meta Prompt Guard 2 仍保留为可选实现；恢复访问后只需切换 YAML 的 `model.implementation`、`name_or_path` 和 `revision`。
