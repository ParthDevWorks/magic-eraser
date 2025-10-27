from abc import abstractmethod

import torch
from magic_eraser.models.ocr.utils.ocr_output import OCRResult
from magic_eraser.models.base import BaseModel


class OCRModel(BaseModel):
    """Abstract Class for All OCR model."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def inference(self, image_tensor: torch.Tensor) -> list[OCRResult]:
        pass
