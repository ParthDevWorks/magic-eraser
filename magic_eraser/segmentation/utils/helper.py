from typing import List
import torch
from magic_eraser.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)


def get_segmentation_masks(
    og_image: torch.Tensor,
    segmentation_output: SegmentationOutput,
    confidence_threshold: float = 0.85,
) -> torch.Tensor:
    """Applies segmentation masks to an original image based on confidence scores.

    Args:
        og_image (torch.Tensor): The original image tensor where segmentation masks will be applied.
        segmentation_output (SegmentationOutput): An instance of SegmentationOutput containing prediction masks and confidence scores.
        confidence_threshold (float): The minimum confidence score required for a mask to be considered for application. Defaults to 0.8

    Returns:
        torch.Tensor: A tensor representing the original image with segmentation masks applied according to the confidence thresholds.
    """
    masks = []
    segmentation_masks = torch.zeros_like(og_image, dtype=torch.bool)

    for score, mask in zip(
        segmentation_output.confidence_scores, segmentation_output.prediction_masks
    ):
        if score > confidence_threshold:
            mask_bool = (mask > 0.5).type(torch.bool)
            masks.append(mask_bool)

    for mask in masks:
        segmentation_masks |= mask

    return segmentation_masks


def filter_mask(
    segmentation_output: SegmentationOutput, target_labels: List[str]
) -> SegmentationOutput:
    """
    Filters out target object regions from the segmentation output.

    Args:
        segmentation_output (SegmentationOutput): Segmentation output containing mask regions.
        target_labels (List[str]): List of target object string labels to filter from the segmentation output.

    Returns:
        SegmentationOutput: Segmentation output containing mask regions for target object regions only.
    """

    label_index = []

    for index, label in enumerate(segmentation_output.labels):
        if label in target_labels:
            label_index.append(index)

    labels = [segmentation_output.labels[i] for i in label_index]
    scores = segmentation_output.confidence_scores[label_index]
    prediction_masks = segmentation_output.prediction_masks[label_index]

    return SegmentationOutput(
        labels=labels,
        confidence_scores=scores,
        prediction_masks=prediction_masks,
    )
