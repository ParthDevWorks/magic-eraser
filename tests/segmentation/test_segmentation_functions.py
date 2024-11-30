import torch
from magic_eraser.segmentation.utils.helper import get_segmentation_masks


def test_get_segmentation_masks(maskrcnn_model_initialized):
    dummy_image = torch.rand(3, 1000, 1000)
    output = maskrcnn_model_initialized.inference(image_tensor=dummy_image)

    masks = get_segmentation_masks(dummy_image, output)

    assert isinstance(masks, torch.Tensor)
    assert masks.dtype == torch.bool
    assert masks.shape == dummy_image.shape
