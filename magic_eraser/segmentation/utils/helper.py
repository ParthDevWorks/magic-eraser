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
