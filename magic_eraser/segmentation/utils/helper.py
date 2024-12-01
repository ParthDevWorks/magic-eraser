import torch
from magic_eraser.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)

HUMAN_LABEL_VALUES = ["person"]


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


def filter_humans_mask(segmentation_output: SegmentationOutput) -> SegmentationOutput:

    human_label_index = []

    for index, label in enumerate(segmentation_output.labels):
        if label in HUMAN_LABEL_VALUES:
            human_label_index.append(index)

    bbox = segmentation_output.bounding_box[human_label_index]
    labels = [segmentation_output.labels[i] for i in human_label_index]
    scores = segmentation_output.confidence_scores[human_label_index]
    prediction_masks = segmentation_output.prediction_masks[human_label_index]

    return SegmentationOutput(
        bounding_box=bbox,
        labels=labels,
        confidence_scores=scores,
        prediction_masks=prediction_masks,
    )
