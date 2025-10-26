import torch

from magic_eraser.segmentation.base import SegmentationModel
from magic_eraser.segmentation.utils.segmentation_output import SegmentationOutput
from magic_eraser.utils.image import dilate_boolean_tensors
from magic_eraser.segmentation.utils.helper import (
    get_segmentation_masks,
    filter_mask,
)
from magic_eraser.inpainting.base import InpaintingModel
from magic_eraser.ocr.base import OCRModel
from magic_eraser.ocr.utils.ocr_output import OCRResult


def perform_segmentation(
    image_tensor: torch.Tensor, segmentation_model: SegmentationModel
) -> list[SegmentationOutput]:
    """
    Performs segmentation on the given image tensor using the provided segmentation model.

    Args:
        image_tensor (torch.Tensor): The input image tensor to perform segmentation on.
        segmentation_model (SegmentationModel): An instance of a SegmentationModel subclass.

    Returns:
        List (SegmentationOutput): A list containing the segmentation results.

    Raises:
        AssertionError: If the segmentation model is None.

    """
    assert segmentation_model is not None

    segmentation_output = segmentation_model.inference(image_tensor=image_tensor)

    return segmentation_output


def perform_inpainting(
    image_tensor: torch.Tensor, masks: torch.Tensor, inpainting_model: InpaintingModel
) -> torch.Tensor:
    """
    Performs inpainting on the given image tensor using the provided inpainting model and masks.

    Args:
        image_tensor (torch.Tensor): The input image tensor to be inpainted.
        masks (torch.Tensor): A tensor containing boolean masks indicating which regions need inpainting.
            These masks should have the same shape as the image_tensor but with True values where inpainting is needed.
        inpainting_model (InpaintingModel): An instance of an InpaintingModel subclass.

    Returns:
        torch.Tensor: The inpainted image tensor.
            This output has the same shape as the input image_tensor, but with the masked regions replaced by inpainted content.

    Raises:
        AssertionError: If the inpainting model is None or not properly configured.
    """
    assert inpainting_model is not None

    inpainted_image = inpainting_model.inference(image=image_tensor, mask=masks)

    return inpainted_image


def perform_ocr(image_tensor: torch.Tensor, ocr_model: OCRModel) -> list[OCRResult]:
    """
    Performs OCR on the given image tensor using the provided ocr model.

    Args:
        image_tensor (torch.Tensor): The input image tensor to perform ocr on.
        ocr_model (OCRModel): An instance of a OCRModel subclass.

    Returns:
        List[OCRResult]: A list of ocr results.

    Raises:
        AssertionError: If the ocr model is None.

    """
    assert ocr_model is not None

    ocr_output = ocr_model.inference(image_tensor=image_tensor)
    return ocr_output


def post_process_mask(
    image_tensor: torch.Tensor,
    segmentation_output: list[SegmentationOutput],
    filter_labels: list[str] | None = None,
    dilate_tensors: bool = True,
) -> torch.Tensor:
    """
    Post-processes the segmentation output to obtain masks for the target object regions only.
    This function filters out the target object regions from the segmentation output if filter_labels provided and dilates the resulting masks.
    The output tensor is of type boolean with shape as (C, H, W).

    Args:
        image_tensor (torch.Tensor): Original image tensor.
        segmentation_output (List(SegmentationOutput)): A list of Segmentation output.
        filter_labels (list[str] | None): List of target object labels to filter from the segmentation output.
        dilate_tensors (bool): Whether to dilate the resulting masks. Defaults to True.

    Returns:
        torch.Tensor: Dilated masks for the target object regions only.

    """
    segmentation_output_with_targets_only = filter_mask(
        segmentation_output, filter_labels
    )

    segmentation_masks = get_segmentation_masks(
        og_image=image_tensor, segmentation_output=segmentation_output_with_targets_only
    )

    if dilate_tensors:
        segmentation_masks = dilate_boolean_tensors(segmentation_masks)

    return segmentation_masks
