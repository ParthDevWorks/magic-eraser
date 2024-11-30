from abc import ABC, abstractmethod

import torch
from magic_eraser.segmentation.utils.segmentation_output import SegmentationOutput


class SegmentationModel(ABC):
    """Abstract Class for All Segmentation model."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def inference(self, image_tensor: torch.Tensor) -> SegmentationOutput:
        pass
