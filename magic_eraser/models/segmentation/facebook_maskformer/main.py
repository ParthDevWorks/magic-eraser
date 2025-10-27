import os
import logging

import torch
from transformers.image_processing_utils import BatchFeature
from transformers import (
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)

from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerForInstanceSegmentationOutput,
)

from magic_eraser.models.segmentation.base import SegmentationModel
from magic_eraser.models.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)
from magic_eraser.models.segmentation.utils.facebook_maskformer.labels import LABELS


class FacebookMaskFormer(SegmentationModel):
    """
    Loads the pretrained MaskFormer model weights.

    Raises:
        ValueError:  If the path to the model weights is not found.
    """

    def __init__(self) -> None:

        SEGMENTATION_MODEL_PATH = os.environ.get("SEGMENTATION_MODEL_PATH")
        if SEGMENTATION_MODEL_PATH is None:
            raise ValueError("SEGMENTATION_MODEL_PATH environment variable is not set")

        self.maskformer_path = os.path.join(
            SEGMENTATION_MODEL_PATH,
            "maskformer",
            "snapshots",
            "cdaa4f1cc221ac6fbc0e42254c5ad13a4a15891c",
        )

        if not os.path.exists(self.maskformer_path):
            raise ValueError(
                f"MaskFormer Model Weights not located inside {SEGMENTATION_MODEL_PATH}"
            )

        logging.info("Loading MaskFormer Model weights")

        # TODO: Figure out why with GPU, the inference time is increasing.
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.feature_extractor = MaskFormerImageProcessor.from_pretrained(
            self.maskformer_path,
        )

        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            self.maskformer_path
        )

        self.model.eval()

    def get_model_id(self) -> str:
        """Returns the identifier for the segmentation model.

        Returns:
            str: A string representing the model ID.
        """
        return "maskformer"

    def _pre_process(self, image: torch.Tensor) -> BatchFeature:
        inputs = self.feature_extractor(
            images=image, return_tensors="pt", do_rescale=False, do_resize=False
        )
        return inputs

    def _post_process(
        self, outputs: MaskFormerForInstanceSegmentationOutput, shape: tuple[int, int]
    ) -> list[SegmentationOutput]:
        """Processes the raw model predictions into a structured format.

        Args:
            outputs (MaskFormerForInstanceSegmentationOutput): The raw predictions from the MaskFormer model.
            shape (int,int): Image Shape

        Returns:
            List (SegmentationOutput): A structured representation of the segmentation results.
        """

        result = self.feature_extractor.post_process_instance_segmentation(
            outputs=outputs, target_sizes=[shape], return_binary_maps=True
        )[0]

        segmentation_output = []

        for segment_info, mask in zip(result["segments_info"], result["segmentation"]):

            segmentation_output.append(
                SegmentationOutput(
                    label=LABELS[segment_info["label_id"]],
                    confidence_score=segment_info["score"],
                    prediction_mask=mask,
                )
            )

        return segmentation_output

    def inference(self, image_tensor: torch.Tensor) -> list[SegmentationOutput]:
        """Performs inference on a image.

        Args:
            image_tensors (torch.Tensor): A tensors representing the image.

        Returns:
            List[SegmentationOutput]
        """

        if self.model is None:
            raise ValueError(
                "Model is not initialized. Initialize the model before calling inference."
            )

        inputs = self._pre_process(image_tensor)
        _, _, height, width = inputs["pixel_values"].shape

        with torch.no_grad():
            outputs: MaskFormerForInstanceSegmentationOutput = self.model(**inputs)

        final_output = self._post_process(outputs, (height, width))

        return final_output

    def shutdown(self) -> None:
        """Cleans up resources by setting the model to None."""
        self.model = None
        self.device = None


# CLI Testing Purpose
if __name__ == "__main__":  # pragma: no cover
    from magic_eraser.utils.image import load_image

    input_path = "/Users/parthrathod/Documents/Projects/MAGIC_ERASER/magic-eraser/sample_data/segmentation/sample_1.jpg"

    image_tensor = load_image(input_path)

    model = FacebookMaskFormer()

    output = model.inference(image_tensor=image_tensor)
    print(output)
