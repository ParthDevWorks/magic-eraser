from abc import ABC, abstractmethod

import torch
from magic_eraser.ocr.utils.ocr_output import OCRResult


class OCRModel(ABC):
    """Abstract Class for All OCR model."""

    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def inference(self, image_tensor: torch.Tensor) -> OCRResult:
        pass
