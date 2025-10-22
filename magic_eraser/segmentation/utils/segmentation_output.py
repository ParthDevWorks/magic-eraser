from dataclasses import dataclass

import torch


@dataclass
class SegmentationOutput:
    labels: list
    confidence_scores: torch.Tensor
    prediction_masks: torch.Tensor
