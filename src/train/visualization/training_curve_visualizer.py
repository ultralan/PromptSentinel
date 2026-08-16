"""根据 Transformers Trainer 状态绘制训练曲线。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingCurveVisualizer:
    """只读取单次训练运行，生成可追溯的损失曲线。"""

    _CHINESE_FONT_CANDIDATES = (
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "Noto Sans CJK SC",
        "SimHei",
    )

    def render(self, run_dir: Path) -> Path:
        """从 trainer_state.json 提取训练与验证 loss 并写入图像和元数据。"""
        state_path = run_dir / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"未找到 Trainer 状态文件: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        train_points, validation_points = self._extract_loss_points(state)
        if not train_points:
            raise ValueError(f"Trainer 状态不包含训练 loss: {state_path}")

        visualization_dir = run_dir / "visualizations"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        output_path = visualization_dir / "loss_curve.png"
        self._plot(train_points, validation_points, output_path)
        (visualization_dir / "loss_curve.json").write_text(
            json.dumps(
                {
                    "source": str(state_path),
                    "train_points": len(train_points),
                    "validation_points": len(validation_points),
                    "best_metric": state.get("best_metric"),
                    "best_model_checkpoint": state.get("best_model_checkpoint"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _extract_loss_points(state: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """将 Trainer 原始 log_history 转为图表需要的规范点位。"""
        train_points, validation_points = [], []
        for entry in state.get("log_history", []):
            if "loss" in entry and "step" in entry:
                train_points.append({"step": float(entry["step"]), "epoch": float(entry["epoch"]), "loss": float(entry["loss"])})
            if "eval_loss" in entry and "step" in entry:
                validation_points.append(
                    {"step": float(entry["step"]), "epoch": float(entry["epoch"]), "loss": float(entry["eval_loss"])}
                )
        return train_points, validation_points

    @staticmethod
    def _plot(train_points: list[dict[str, float]], validation_points: list[dict[str, float]], output_path: Path) -> None:
        """绘制原始训练 loss、滑动均值和按 epoch 记录的验证 loss。"""
        steps = [point["step"] for point in train_points]
        losses = [point["loss"] for point in train_points]
        window = min(20, max(1, len(losses) // 20))
        moving_average = [sum(losses[max(0, index - window + 1) : index + 1]) / min(index + 1, window) for index in range(len(losses))]

        font = TrainingCurveVisualizer._resolve_chinese_font()
        figure, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
        axis.plot(steps, losses, color="#92A8D1", alpha=0.35, linewidth=1, label="训练 loss（原始）")
        axis.plot(steps, moving_average, color="#1F4E79", linewidth=2, label=f"训练 loss（{window} 点滑动均值）")
        if validation_points:
            axis.plot(
                [point["step"] for point in validation_points],
                [point["loss"] for point in validation_points],
                color="#C55A11",
                marker="o",
                linewidth=2,
                label="验证 loss",
            )
        axis.set_title("训练与验证损失", fontproperties=font)
        axis.set_xlabel("训练 step", fontproperties=font)
        axis.set_ylabel("交叉熵损失", fontproperties=font)
        axis.grid(alpha=0.25)
        axis.legend(prop=font)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

    @classmethod
    def _resolve_chinese_font(cls) -> font_manager.FontProperties:
        """按系统已安装字体选择中文字体，避免依赖特定操作系统路径。"""
        installed_font_names = {font.name for font in font_manager.fontManager.ttflist}
        for font_name in cls._CHINESE_FONT_CANDIDATES:
            if font_name in installed_font_names:
                return font_manager.FontProperties(family=font_name)
        return font_manager.FontProperties()
