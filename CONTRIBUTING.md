# 贡献指南

感谢对 PromptSentinel 的关注。项目聚焦于可复现的提示注入旁路检测实验；任何贡献都应保留实验口径、数据隔离和结果可追溯性。

## 开始前

1. 先在 Issue 中说明要解决的问题、预期影响和验证方式；较大的训练或评测改动请先讨论数据、指标和计算资源。
2. 使用 Python 3.12 与 `uv sync` 安装依赖。
3. 不提交 `data/raw/`、`data/prepared/`、`outputs/`、模型权重、日志、访问令牌或其他凭据。

## 开发约定

- `core` 只放跨模块稳定的实体、枚举与配置 dataclass；`preprocess` 负责原始数据到 prepared 数据的转换；`train` 和 `evaluate` 只消费 prepared 数据。
- 新增数据源时固定上游 revision，记录 split 行数、标签分布与文本 SHA-256，并检查跨 split 重叠。
- 新增模型或训练策略时，保持 Base、LoRA、Full FT 的比较口径可追溯；不得使用 test 做 checkpoint 选择、阈值选择或超参数搜索。
- 代码、配置和文档修改应保持范围小，并为非显而易见的逻辑添加中文注释。

## 提交前检查

```bash
uv run python -m compileall -q src
git diff --check
```

涉及训练或评测语义的改动，还应在 Pull Request 中说明：数据 revision、配置差异、随机种子、运行环境、是否复用既有 checkpoint，以及可复现的指标产物位置。

## Pull Request

请使用仓库模板，描述问题、方案、验证结果与风险。不要将单次随机种子、未冻结测试集或跨口径指标描述为稳定结论。
