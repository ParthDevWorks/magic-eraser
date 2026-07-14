from abc import abstractmethod

import torch

from magic_eraser.models.base import BaseModel


class InpaintingModel(BaseModel):
    """Abstract Class for All Inpainting model."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def inference(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass
