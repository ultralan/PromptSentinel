# 数据目录

`data/raw/`：由 `src.train.prepare_data` 下载的 PromptShield 原始 split，不提交 Git。

`data/prepared/`：经过字段校验与长度过滤后供训练读取的 JSONL manifest，不提交 Git。

`data/reports/`：数据 revision、SHA-256、标签分布和 token 长度审计报告，不提交 Git。

`src/core/data_contracts.py` 定义 `text_classification/v1`：每行固定为 `id`、`text`、`label`、`token_length`。`src/preprocess` 负责生成该接口，训练层只读取该接口，不读取原始文件，也不依赖数据集的字段名或下载方式。

训练代码不读取 test split；test split 仅在后续 `src/evaluate/` 中使用。
