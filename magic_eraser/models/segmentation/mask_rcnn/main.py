import logging
import os

import torch
from magic_eraser.models.segmentation.base import SegmentationModel
from magic_eraser.models.segmentation.utils.mask_rcnn.coco_dataset_classnames import (
    COCO_DATASET_CLASSNAMES,
)
from magic_eraser.models.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)
from torchvision.models.detection import maskrcnn_resnet50_fpn


class MaskRcnn(SegmentationModel):
    """
    Loads the pretrained Mask R-CNN model weights.

    Raises:
        ValueError:  If the path to the model weights is not found.
    """

    def __init__(self) -> None:

        SEGMENTATION_MODEL_PATH = os.environ.get("SEGMENTATION_MODEL_PATH")
        if SEGMENTATION_MODEL_PATH is None:
            raise ValueError("SEGMENTATION_MODEL_PATH environment variable is not set")

        mask_rcnn_path = os.path.join(
            SEGMENTATION_MODEL_PATH,
            "mask_rcnn",
            "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        )

        if not os.path.exists(mask_rcnn_path):
            raise ValueError(
                f"Mask RCNN Model Weights not located inside {SEGMENTATION_MODEL_PATH}"
            )

        logging.info("Loading Mask R-CNN model weights")
        self.model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
        )
        self.model.load_state_dict(torch.load(mask_rcnn_path))

        self.model.eval()  # Set the Model to Evaluation Mode

    def get_model_id(self) -> str:
        """Returns the identifier for the segmentation model.

        Returns:
            str: A string representing the model ID.
        """
        return "mask_rcnn"

    def inference(self, image_tensor: torch.Tensor) -> list[SegmentationOutput]:
        """Performs inference on a image.

        Args:
            image_tensors (torch.Tensor): A tensors representing the image.

        Returns:
            List (SegmentationOutput)
        """
        if self.model is None:
            raise ValueError("Mask R-CNN model has not been initialized")

        with torch.no_grad():
            pred = self.model([image_tensor])[0]

        post_processing_output = self._post_processing(pred)

        return post_processing_output

    def _post_processing(self, prediction) -> list[SegmentationOutput]:
        """Processes the raw model predictions into a structured format.

        Args:
            predictions (_type_): The raw predictions from the Mask R-CNN model.

        Returns:
            List (SegmentationOutput): A structured representation of the segmentation results.
        """

        output = []

        for label, score, mask in zip(
            prediction["labels"].cpu().numpy().tolist(),
            prediction["scores"],
            prediction["masks"],
        ):
            output.append(
                SegmentationOutput(
                    label=COCO_DATASET_CLASSNAMES.get(label, "undefined"),
                    confidence_score=score.item(),
                    prediction_mask=mask,
                )
            )

        return output

    def shutdown(self) -> None:
        """Cleans up resources by setting the model to None."""
        self.model = None


# CLI Testing Purpose
if __name__ == "__main__":
    from magic_eraser.utils.image import load_image

    input_path = "/Users/parthrathod/Documents/Projects/MAGIC_ERASER/magic-eraser/sample_data/segmentation/sample_1.jpg"

    image_tensor = load_image(input_path)

    model = MaskRcnn()

    output = model.inference(image_tensor=image_tensor)
    print(output)
