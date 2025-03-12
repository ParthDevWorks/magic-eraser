from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class InpaintingModel(ABC):
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
