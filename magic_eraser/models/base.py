from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Class for All Magic Eraser models."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass
