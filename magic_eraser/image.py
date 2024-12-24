import torch
from magic_eraser.model_initialization.initialize import ModelInitializer
from magic_eraser.segmentation.base import SegmentationModel
from magic_eraser.segmentation.utils.segmentation_output import SegmentationOutput
from magic_eraser.utils.image import dilate_boolean_tensors, rgb_to_grayscale
from magic_eraser.segmentation.utils.helper import (
    get_segmentation_masks,
    filter_humans_mask,
)
from magic_eraser.inpainting.base import InpaintingModel


def perform_segmentation(
    image_tensor: torch.Tensor, segmentation_model: SegmentationModel
) -> SegmentationOutput:
    """
    Performs segmentation on the given image tensor using the provided segmentation model.

    Args:
        image_tensor (torch.Tensor): The input image tensor to perform segmentation on.
        segmentation_model (SegmentationModel): An instance of a SegmentationModel subclass.

    Returns:
        SegmentationOutput: An object containing the segmentation results.

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


def remove_humans(
    image_tensor: torch.Tensor, eraser_config: ModelInitializer
) -> torch.Tensor:
    """
    Removes humans from an input image tensor using a combination of segmentation and inpainting models.

    This function performs two main steps:
    1. Segments the human regions in the input image using a pre-trained segmentation model.
    2. Inpaints the segmented human regions using a pre-trained inpainting model.

    Args:
        image_tensor (torch.Tensor): The input image tensor to process.
        eraser_config (ModelInitializer): A configuration object containing settings for both segmentation and inpainting models.

    Returns:
        torch.Tensor: The processed image tensor with humans removed from the original input.

    Raises:
        AssertionError: If either the segmentation or inpainting model is not properly configured in the ModelInitializer object.

    Notes:
        - This function assumes that the ModelInitializer object has been properly initialized with valid model paths and parameters.
        - The function uses the models specified in the ModelInitializer object for both segmentation and inpainting tasks.
        - The output tensor will have the same shape as the input tensor, but with the human regions replaced by inpainted content.

    """

    segmentation_model = eraser_config.get_segmentation_model()

    inpainting_model = eraser_config.get_inpainting_model()

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_humans_mask(image_tensor, segmentation_output)

    inpainted_image = perform_inpainting(
        image_tensor, dilated_segmented_mask, inpainting_model
    )

    return inpainted_image


def color_splash_humans(
    image_tensor: torch.Tensor, eraser_config: ModelInitializer
) -> torch.Tensor:

    segmentation_model = eraser_config.get_segmentation_model()

    segmentation_output = perform_segmentation(image_tensor, segmentation_model)

    dilated_segmented_mask = post_process_humans_mask(
        image_tensor, segmentation_output, dilate_tensors=False
    )

    if dilated_segmented_mask.dtype != "float32":
        dilated_segmented_mask = dilated_segmented_mask.float()

    og_grayscale_image = rgb_to_grayscale(image_tensor)

    color_splash_tensor = (
        dilated_segmented_mask * image_tensor
        + (1 - dilated_segmented_mask) * og_grayscale_image
    )

    return color_splash_tensor


def post_process_humans_mask(
    image_tensor: torch.Tensor,
    segmentation_output: SegmentationOutput,
    dilate_tensors: bool = True,
) -> torch.Tensor:
    """
    Post-processes the segmentation output to obtain masks for the human regions only.

    This function filters out the human regions from the segmentation output and dilates the resulting masks.

    Args:
        image_tensor (torch.Tensor): Original image tensor.
        segmentation_output (SegmentationOutput): Segmentation output containing mask regions.
        dilate_tensors (bool): Whether to dilate the resulting masks. Defaults to True.

    Returns:
        torch.Tensor: Dilated masks for the human regions only.

    """
    segmentation_output_with_humans_only = filter_humans_mask(segmentation_output)

    segmentation_masks = get_segmentation_masks(
        og_image=image_tensor, segmentation_output=segmentation_output_with_humans_only
    )

    if dilate_tensors:
        segmentation_masks = dilate_boolean_tensors(segmentation_masks)

    return segmentation_masks
