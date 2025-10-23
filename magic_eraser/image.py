import torch
import scipy

from magic_eraser.segmentation.base import SegmentationModel
from magic_eraser.segmentation.utils.segmentation_output import SegmentationOutput
from magic_eraser.utils.image import dilate_boolean_tensors, rgb_to_grayscale
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


def erase(
    image_tensor: torch.Tensor,
    segmentation_model: SegmentationModel,
    inpainting_model: InpaintingModel,
    target_labels: list[str],
) -> torch.Tensor:
    """
    Erases target objects defined in config from an input image tensor using a combination of segmentation and inpainting models.

    This function performs two main steps:
    1. Segments the target object regions in the input image using a pre-trained segmentation model.
    2. Inpaints the segmented target object regions using a pre-trained inpainting model.

    Args:
        image_tensor (torch.Tensor): The input image tensor to process.
        segmentation_model (SegmentationModel): An instance of a SegmentationModel subclass.
        inpainting_model (InpaintingModel): An instance of an InpaintingModel subclass.
        target_labels (list[str]): A list of target object labels to remove from the input image.

    Returns:
        torch.Tensor: The processed image tensor with target object removed from the original input.

    Raises:
        AssertionError: If either the segmentation or inpainting model is not properly configured in the ModelInitializer object.

    Notes:
        - This function assumes that the ModelInitializer object has been properly initialized with valid model paths and parameters.
        - The function uses the models specified in the ModelInitializer object for both segmentation and inpainting tasks.
        - The output tensor will have the same shape as the input tensor, but with the target object regions replaced by inpainted content.

    """

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        target_labels=target_labels,
    )
    if dilated_segmented_mask.any():
        inpainted_image = perform_inpainting(
            image_tensor, dilated_segmented_mask, inpainting_model
        )

        return inpainted_image
    else:
        return image_tensor


def color_splash(
    image_tensor: torch.Tensor,
    segmentation_model: SegmentationModel,
    target_labels: list[str],
) -> torch.Tensor:
    """
    Applies a color splash effect to target objct regions in an input image tensor.

    Args:
        image_tensor (torch.Tensor): The input image tensor to process.
        segmentation_model (SegmentationModel): An instance of a SegmentationModel subclass.
        target_labels (list[str]): A list of target object labels to color splash from the input image.

    Returns:
        torch.Tensor: The processed image tensor with a color splash effect applied to target object regions.

    Raises:
        AssertionError: If the segmentation model is not properly configured in the ModelInitializer object.
    """

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        target_labels=target_labels,
        dilate_tensors=False,
    )

    if dilated_segmented_mask.dtype != "float32":
        dilated_segmented_mask = dilated_segmented_mask.float()

    og_grayscale_image = rgb_to_grayscale(image_tensor)

    color_splash_tensor = (
        dilated_segmented_mask * image_tensor
        + (1 - dilated_segmented_mask) * og_grayscale_image
    )

    return color_splash_tensor


def remove_background(
    image_tensor: torch.Tensor,
    segmentation_model: SegmentationModel,
    target_labels: list[str],
) -> torch.Tensor:
    """
    Removes the background from an input image tensor using a combination of segmentation and masking techniques.

    Args:
        image_tensor (torch.Tensor): The input image tensor to process.
        segmentation_model (SegmentationModel): An instance of a SegmentationModel subclass.
        target_labels (list[str]): A list of target object labels to keep from the input image.

    Returns:
        torch.Tensor: The processed image tensor with target kept and the background removed.

    Raises:
        AssertionError: If the segmentation model is not properly configured in the ModelInitializer object.

    """

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        target_labels=target_labels,
        dilate_tensors=False,
    )

    if dilated_segmented_mask.dtype != "float32":
        dilated_segmented_mask = dilated_segmented_mask.float()

    background_removed_tensor = dilated_segmented_mask * image_tensor + (
        1 - dilated_segmented_mask
    ) * torch.ones(image_tensor.shape[-2:])

    return background_removed_tensor


def remove_text(image_tensor: torch.Tensor, ocr_model: OCRModel) -> torch.Tensor:
    """
    Removes the text from an input image tensor using ocr model and masking techniques.

    Args:
        image_tensor (torch.Tensor): The input image tensor to process.
        ocr_model (SegmentationModel): An instance of a OCRModel subclass.

    Returns:
        torch.Tensor: The processed image tensor with text removed.

    Raises:
        AssertionError: If the ocr model is not properly configured in the ModelInitializer object.

    """
    ocr_output: list[OCRResult] = perform_ocr(image_tensor, ocr_model)

    for output in ocr_output:
        x, y, w, h = output.bounding_box.to_tuple

        roi = image_tensor[:, y : y + h, x : x + w]
        roi_int = (roi * 255).to(torch.int)

        # Flatten and compute mode per channel
        mode_values = []
        for c in range(roi_int.shape[0]):  # Iterate over channels
            mode_value = scipy.stats.mode(
                roi_int[c].flatten().numpy(), keepdims=False
            ).mode
            mode_values.append(mode_value / 255.0)  # Convert back to float [0,1]

        mode_tensor = torch.tensor(mode_values, dtype=torch.float32).view(3, 1, 1)

        image_tensor[:, y : y + h, x : x + w] = mode_tensor

    return image_tensor


def post_process_mask(
    image_tensor: torch.Tensor,
    segmentation_output: list[SegmentationOutput],
    target_labels: list,
    dilate_tensors: bool = True,
) -> torch.Tensor:
    """
    Post-processes the segmentation output to obtain masks for the target object regions only.

    This function filters out the target object regions from the segmentation output and dilates the resulting masks.

    Args:
        image_tensor (torch.Tensor): Original image tensor.
        segmentation_output (List(SegmentationOutput)): A list of Segmentation output.
        target_labels (list): List of target object labels to filter from the segmentation output.
        dilate_tensors (bool): Whether to dilate the resulting masks. Defaults to True.

    Returns:
        torch.Tensor: Dilated masks for the target object regions only.

    """
    segmentation_output_with_targets_only = filter_mask(
        segmentation_output, target_labels
    )

    segmentation_masks = get_segmentation_masks(
        og_image=image_tensor, segmentation_output=segmentation_output_with_targets_only
    )

    if dilate_tensors:
        segmentation_masks = dilate_boolean_tensors(segmentation_masks)

    return segmentation_masks
