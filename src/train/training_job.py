"""只消费 prepared 数据集的训练业务编排。"""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from src.core.config_entity import DataConfig, TrainingConfig
from src.core.data_entity import DatasetSplit
from src.preprocess.prepared_dataset import PreparedDataset
from src.train.fine_tuning.fine_tuning_strategy import FineTuningStrategy
from src.train.model_trainer.sequence_classification_trainer import SequenceClassificationTrainer
from src.train.training_run import TrainingRun


class TrainingJob:
    """消费 prepared 数据集并委托策略完成一次训练任务。"""

    def __init__(
        self,
        config: TrainingConfig,
        data_config: DataConfig,
        prepared_dataset: PreparedDataset,
        model_trainer: SequenceClassificationTrainer,
        strategy: FineTuningStrategy,
        training_run: TrainingRun,
    ) -> None:
        """注入训练配置、数据集、模型训练器、策略和运行产物对象。"""
        self._config = config
        self._data_config = data_config
        self._prepared_dataset = prepared_dataset
        self._model_trainer = model_trainer
        self._strategy = strategy
        self._training_run = training_run

    def run(self) -> None:
        """校验 prepared 契约，训练模型并保存策略定义的产物。"""
        manifest = self._prepared_dataset.load_manifest()
        if manifest.max_length != self._config.model.max_length:
            raise ValueError("prepared manifest 与训练模型的 max_length 不一致")
        TrainingRun.set_seed(self._config.run.seed)
        tokenizer = self._model_trainer.load_tokenizer()
        base_model = self._model_trainer.load_base_model()
        model = self._strategy.prepare_model(base_model)
        train_dataset = self._to_hf_dataset(self._prepared_dataset.load_training_split(DatasetSplit.TRAIN), tokenizer)
        validation_dataset = self._to_hf_dataset(self._prepared_dataset.load_training_split(DatasetSplit.VALIDATION), tokenizer)
        device = self._model_trainer.select_device()
        self._training_run.initialize(
            self._data_config,
            self._parameter_summary(model),
            {"train": len(train_dataset), "validation": len(validation_dataset)},
            device,
        )
        trainer = self._model_trainer.create_trainer(
            model, tokenizer, train_dataset, validation_dataset, self._training_run.checkpoint_dir, device,
        )
        trainer.train()
        self._strategy.save_model(trainer, tokenizer, self._training_run.model_dir)
        self._training_run.save_state(trainer)

    def _to_hf_dataset(self, records: list[Any], tokenizer: Any) -> Dataset:
        """将标准分类记录 token 化为 Transformers Trainer 可消费的数据集。"""
        dataset = Dataset.from_list([record.to_dict() for record in records])
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=self._config.model.max_length),
            batched=True,
            remove_columns=["id", "text", "token_length"],
        )

    @staticmethod
    def _parameter_summary(model: Any) -> dict[str, int]:
        """统计模型总参数量与当前策略实际可训练的参数量。"""
        return {
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "total": sum(parameter.numel() for parameter in model.parameters()),
        }
