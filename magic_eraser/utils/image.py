from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torchvision.transforms import ToPILImage, transforms
from PIL import Image
from pillow_heif import register_heif_opener
import torchvision.transforms as T


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


def apply_color_splash(og_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies color splash effect to an image based on a given mask.

    This function takes an original image tensor and a mask tensor, applies a color splash effect
    by combining the original image colors with a grayscale representation of the image
    where the mask indicates areas to be kept in full color.

    Args:
        og_image (torch.Tensor): The original image tensor with shape (C, H, W), where C is the number of channels.
        mask (torch.Tensor): The mask tensor with shape (C, H, W), indicating areas to be kept in full color.

    Returns:
        torch.Tensor: The resulting color-splashed image tensor with the same shape as the input.

    Notes:
        - If the mask dtype is not float32, it is converted to float32 for consistent operations.
    """

    # Convert image to grayscale
    grayscale_image = og_image.mean(dim=0, keepdim=True)  # Shape: (1, H, W)
    og_grayscale_image = grayscale_image.repeat(3, 1, 1)  # Convert to (C=3, H, W)

    if mask.dtype != "float32":
        mask = mask.float()

    # Apply mask to combine grayscale and color
    color_splash_image = mask * og_image + (1 - mask) * og_grayscale_image
    return color_splash_image
