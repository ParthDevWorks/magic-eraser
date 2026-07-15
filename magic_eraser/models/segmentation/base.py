from abc import abstractmethod

import torch

from magic_eraser.models.base import BaseModel
from magic_eraser.models.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)


class SegmentationModel(BaseModel):
    """Abstract Class for All Segmentation model."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def inference(self, image_tensor: torch.Tensor) -> list[SegmentationOutput]:
        pass
