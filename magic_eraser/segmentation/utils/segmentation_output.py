from dataclasses import dataclass

import torch


@dataclass
class SegmentationOutput:
    bounding_box: torch.Tensor
    labels: list
    confidence_scores: torch.Tensor
    prediction_masks: torch.Tensor
