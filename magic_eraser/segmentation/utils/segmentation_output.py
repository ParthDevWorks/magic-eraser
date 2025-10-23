from dataclasses import dataclass

import torch


@dataclass
class SegmentationOutput:
    label: str
    confidence_score: float
    prediction_mask: torch.Tensor
