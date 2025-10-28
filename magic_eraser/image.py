import torch
import scipy
import torchvision.transforms.functional as TVF

from magic_eraser.models.segmentation.base import SegmentationModel
from magic_eraser.utils.image import rgb_to_grayscale

from magic_eraser.models.inpainting.base import InpaintingModel
from magic_eraser.models.ocr.base import OCRModel
from magic_eraser.models.ocr.utils.ocr_output import OCRResult
from magic_eraser.helper import (
    perform_segmentation,
    perform_inpainting,
    perform_ocr,
    post_process_mask,
)


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
        AssertionError: If either the segmentation or inpainting model is None.

    Notes:
        - The output tensor will have the same shape as the input tensor, but with the target object regions replaced by inpainted content.

    """

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        filter_labels=target_labels,
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
        AssertionError: If the segmentation model is None.
    """

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        filter_labels=target_labels,
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
        AssertionError: If the segmentation model is None.

    """

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        filter_labels=target_labels,
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
        AssertionError: If the ocr model is None.

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


def fall_color(
    image_tensor: torch.Tensor, segmentation_model: SegmentationModel
) -> torch.Tensor:
    """
    Applies a fall color effect to tree regions in an input image tensor.

    This function performs the following steps:
    1. Performs segmentation on the input image using the provided segmentation model.
    2. Post-processes the segmentation output to obtain masks for tree regions.
    3. Applies a specific hue and saturation to the identified tree regions.

    Args:
        image_tensor (torch.Tensor): The input image tensor to process.
        segmentation_model (SegmentationModel): An instance of a SegmentationModel subclass.

    Returns:
        torch.Tensor: The processed image tensor with fall color applied to tree regions.

    Raises:
        AssertionError: If the segmentation model is None.
    """
    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_mask(
        image_tensor,
        segmentation_output,
        filter_labels=["tree"],
        dilate_tensors=False,
    )

    if dilated_segmented_mask.dtype != "float32":
        dilated_segmented_mask = dilated_segmented_mask.float()

    non_masked_colored_regions = (1 - dilated_segmented_mask) * image_tensor
    masked_colored_regions = dilated_segmented_mask * image_tensor

    img_filtered = TVF.adjust_hue(masked_colored_regions, -0.1)
    masked_color_enhance_regions = TVF.adjust_saturation(img_filtered, 1.5)

    final_img = non_masked_colored_regions + masked_color_enhance_regions

    return final_img
