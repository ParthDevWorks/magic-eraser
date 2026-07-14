from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pillow_heif import register_heif_opener
from scipy.ndimage import binary_dilation
from torchvision.transforms import ToPILImage, transforms


def load_image(path: Path) -> torch.Tensor:
    """Load Image as tensors from a given path

    Args:
        path (Path): String like path

    Returns:
        torch.Tensor: List of Image Tensors
    """

    # HEIC file formats are not supported by opencv-python as of 4.9.0.80 version
    if path.split(".")[-1].lower() == "heic":
        register_heif_opener()
        image = Image.open(path)
    else:
        image = Image.open(path)
    image_tensor = transforms.ToTensor()(image)

    return image_tensor


def resize_image(image: torch.Tensor, target_size=(3000, 4000)):
    """
    Resize a image to a common size.

    Args:
    - image: A Tensor object.
    - target_size: The desired output size as (height, width).

    Returns:
    - A resized image as Tensor object.
    """

    # Create a transform to resize the images
    resize_transform = T.Resize(target_size)

    # Apply the resize transform to image tensor
    resized_tensor = resize_transform(image)

    return resized_tensor


def save_image(tensor: torch.Tensor, output_path: str, format: str = "PNG") -> None:
    """
    Save a tensor image to a given path in required format.

    Args:
    - tensor: Tensor object representing the image.
    - output_path: The path where the image should be saved.
    - format: The format in which the image should be saved. Defautls to PNG.
    """

    pil_image = ToPILImage()(tensor)
    pil_image.save(output_path, format)
    print(f"Image saved at: {output_path}")


def dilate_boolean_tensors(image: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """
    Expands (dilates) the True values in the first channel of a multi-channel boolean mask
    using a custom kernel with immediate neighbors only (up, down, left, right) for a specified number of iterations.
    The dilated values are then copied to the rest of the channels.

    Args:
        image (torch.Tensor): The original boolean image with shape (C, H, W) where C is the number of channels.
        iterations (int): The number of iterations to perform dilation.

    Returns:
        torch.Tensor: The dilated boolean image with the same shape as the input.
    """
    # Ensure the input is a boolean tensor
    if image.dtype != torch.bool:
        image = image.bool()

    # Create a structuring element for immediate neighbors (4-connectivity)
    kernel = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    )

    # Convert the first channel to a NumPy array for SciPy processing
    first_channel_mask_np = image[0].numpy()

    # Perform binary dilation using SciPy with the custom kernel for the first channel
    dilated_first_channel_np = binary_dilation(
        first_channel_mask_np, structure=kernel, iterations=iterations
    )

    # Convert the dilated first channel back to a boolean tensor
    dilated_first_channel = torch.from_numpy(dilated_first_channel_np).bool()

    # Create the dilated mask by stacking the dilated first channel across all channels
    dilated_mask = dilated_first_channel.unsqueeze(0).expand_as(image)

    return dilated_mask


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to grayscale using perceptual weights.
    This approach uses the luminance formula, which applies different weights to the RGB channels based on human perception:

    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    These weights reflect how humans perceive green (stronger impact), red, and blue (weaker impact) differently.

    Args:
        image (torch.Tensor): RGB image tensor of shape (C, H, W).

    Returns:
        torch.Tensor: Grayscale image tensor of shape (C, H, W).
    """
    # Weights for the RGB channels
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(image.device)
    # Apply weights to the RGB channels and sum them
    grayscale = (image * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)
    grayscale_image = grayscale.repeat(3, 1, 1)
    return grayscale_image
