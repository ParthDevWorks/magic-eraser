import matplotlib.pyplot as plt
import torch
from magic_eraser.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)
from torchvision.utils import draw_bounding_boxes


def plot_image(
    og_image: torch.Tensor = None,
    masked_image: torch.Tensor = None,
    inpainted_image: torch.Tensor = None,
):
    """This function is use to plot image before segmentation and after segmentation.

    Args:
        image (torch.Tensor): Original Image tensors
        masked_image (torch.Tensor): Masked Image tensors
        inpainted_image (torch.Tensor): InPainted Image tensors
    """
    count = 0

    if (
        og_image is not None and og_image.dtype != "uint8"
    ):  # torchvision 0.18 doesnt support torch.float32 for plotting bounding box
        og_image = (og_image * 255).type(torch.uint8)
        count += 1

    if (
        masked_image is not None and masked_image.dtype != "uint8"
    ):  # torchvision 0.18 doesnt support torch.float32 for plotting bounding box
        masked_image = (masked_image * 255).type(torch.uint8)
        count += 1

    if (
        inpainted_image is not None and inpainted_image.dtype != "uint8"
    ):  # torchvision 0.18 doesnt support torch.float32 for plotting bounding box
        inpainted_image = (inpainted_image * 255).type(torch.uint8)
        count += 1

    _, ax = plt.subplots(1, count, figsize=(15, 20))

    if og_image is not None:
        ax[0].imshow(og_image.permute(1, 2, 0))  # MatplotLib expects HWC format
        ax[0].set_title("Original Image")
        ax[0].axis("off")

    # Plotting after image
    if masked_image is not None:
        ax[1].imshow(masked_image.permute(1, 2, 0))  # MatplotLib expects HWC format
        ax[1].set_title("Masked Image")
        ax[1].axis("off")

    if inpainted_image is not None:
        ax[2].imshow(inpainted_image.permute(1, 2, 0))  # MatplotLib expects HWC format
        ax[2].set_title("In-Painted Image")
        ax[2].axis("off")

    plt.tight_layout()
    plt.show()


def _plot_bounding_boxes(
    normalized_image: torch.Tensor,
    output: SegmentationOutput,
    confidence_threshold: float,
) -> torch.Tensor:
    """To plot bounding box on the image

    Args:
        normalized_image (torch.Tensor): Normalized Image Doesnt accept torch.float as of torchvision 0.18
        output (SegmentationOutput): Output of the Segmentation Model
        confidence_threshold (float): Threshold to reject anything below this value.

    Returns:
        torch.Tensor: Tensors with bounding box if present.
    """
    boxes = []
    labels = []

    for box, label, score in zip(
        output.bounding_box, output.labels, output.confidence_scores
    ):
        if score > confidence_threshold:
            boxes.append(box)
            labels.append(label)

    if boxes:
        tensor_boxes = torch.stack(boxes, dim=0)

        bounding_box_tensor = draw_bounding_boxes(
            normalized_image,
            tensor_boxes,
            labels=labels,
            width=5,
        )

    return bounding_box_tensor
