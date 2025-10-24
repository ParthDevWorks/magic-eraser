from typing import List
import torch
from magic_eraser.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)


def get_segmentation_masks(
    og_image: torch.Tensor,
    segmentation_output: List[SegmentationOutput],
    confidence_threshold: float = 0.85,
) -> torch.Tensor:
    """
    Applies segmentation masks to an original image based on confidence scores.
    The output tensor is of type boolean with shape as (C, H, W).

    Args:
        og_image (torch.Tensor): The original image tensor where segmentation masks will be applied.
        segmentation_output (List(SegmentationOutput)): A list of SegmentationOutput containing prediction masks and confidence scores.
        confidence_threshold (float): The minimum confidence score required for a mask to be considered for application. Defaults to 0.8

    Returns:
        torch.Tensor: A tensor representing the original image with segmentation masks applied according to the confidence thresholds.
    """
    masks = []
    segmentation_masks = torch.zeros_like(og_image, dtype=torch.bool)

    for item in segmentation_output:
        if item.confidence_score > confidence_threshold:
            mask_bool = (item.prediction_mask > 0.5).type(torch.bool)
            masks.append(mask_bool)

    for mask in masks:
        segmentation_masks |= mask

    return segmentation_masks


def filter_mask(
    segmentation_output: List[SegmentationOutput], target_labels: List[str]
) -> List[SegmentationOutput]:
    """
    Filters out target object regions from the segmentation output.

    Args:
        segmentation_output (List(SegmentationOutput)): A list of Segmentation output.
        target_labels (List[str]): List of target object string labels to filter from the segmentation output.

    Returns:
        List (SegmentationOutput): A filtered list of Segmentation output containing mask regions for target object regions only.
    """

    filtered_segmentation_output = []

    for item in segmentation_output:
        if item.label in target_labels:
            filtered_segmentation_output.append(item)

    return filtered_segmentation_output
