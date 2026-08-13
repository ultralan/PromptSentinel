"""模型实现标识到训练器对象的唯一装配点。"""

from src.core.config_entity import ModelConfig, TrainerConfig
from src.core.training_entity import ModelImplementation
from src.train.model_trainer.prompt_guard_trainer import PromptGuardTrainer
from src.train.model_trainer.protect_ai_deberta_trainer import ProtectAiDebertaTrainer
from src.train.model_trainer.sequence_classification_trainer import SequenceClassificationTrainer


class ModelTrainerFactory:
    """根据强类型模型实现标识创建对应的序列分类训练器。"""

    @staticmethod
    def create(model_config: ModelConfig, trainer_config: TrainerConfig | None = None) -> SequenceClassificationTrainer:
        """在唯一映射点选择模型专属加载和校验实现。"""
        if model_config.implementation is ModelImplementation.PROMPT_GUARD_2:
            return PromptGuardTrainer(model_config, trainer_config)
        if model_config.implementation is ModelImplementation.PROTECT_AI_DEBERTA_V2:
            return ProtectAiDebertaTrainer(model_config, trainer_config)
        raise ValueError(f"不支持的模型实现: {model_config.implementation}")
