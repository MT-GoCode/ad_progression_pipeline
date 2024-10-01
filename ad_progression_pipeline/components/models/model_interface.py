from abc import ABC, abstractmethod
from typing import Type


class ModelInterface(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def train(self, **kwargs) -> None:  # noqa: ANN003
        pass

    @abstractmethod
    def infer_and_gather_metrics(self, **kwargs) -> dict:  # noqa: ANN003
        pass

    @abstractmethod
    def serialize(self, output_dir: str) -> None:
        pass

    @abstractmethod
    def deserialize(self, input_dir: str) -> None:
        pass


def instantiate_model(model_class: Type[ModelInterface]) -> ModelInterface:
    return model_class()
