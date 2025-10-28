import torch

from magic_eraser.config.config import (
    Config,
    DEFAULT_SEGMENTATION_MODEL_ID,
    DEFAULT_INPAINTING_MODEL_ID,
    DEFAULT_OCR_MODEL_ID,
)
from magic_eraser.models import get_model
from magic_eraser.models.segmentation.base import SegmentationModel
from magic_eraser.models.inpainting.base import InpaintingModel
from magic_eraser.models.ocr.base import OCRModel
from magic_eraser.image import (
    erase,
    color_splash,
    remove_background,
    remove_text,
    fall_color,
)


class Pipeline:
    """
    A class responsible for orchestrating the image processing pipeline.

    Attributes:
        config (Config): An instance of Config for loading configuration.
    """

    def __init__(self, config: Config) -> None:
        self.segmentation_model: SegmentationModel = get_model(
            DEFAULT_SEGMENTATION_MODEL_ID
        )
        self.inpainting_model: InpaintingModel = get_model(DEFAULT_INPAINTING_MODEL_ID)
        self.ocr_model: OCRModel = get_model(DEFAULT_OCR_MODEL_ID)
        self.config = config.model_dump()

    def analyze_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Analyzes the input image based on the configured mode and processes it accordingly.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Processed image tensor.
        """

        self.image_processed_successfully = False

        if self.config["mode"] == "erase":
            output_image_tensor = erase(
                image_tensor,
                self.segmentation_model,
                self.inpainting_model,
                self.config["target"],
            )

        elif self.config["mode"] == "color_splash":
            output_image_tensor = color_splash(
                image_tensor, self.segmentation_model, self.config["target"]
            )

        elif self.config["mode"] == "remove_background":
            output_image_tensor = remove_background(
                image_tensor, self.segmentation_model, self.config["target"]
            )

        elif self.config["mode"] == "remove_text":
            output_image_tensor = remove_text(image_tensor, self.ocr_model)

        elif self.config["mode"] == "fall_color":
            output_image_tensor = fall_color(image_tensor, self.segmentation_model)

        self.image_processed_successfully = True
        return output_image_tensor

    @property
    def is_image_processing_successfull(self) -> bool:
        """
        Checks if the image processing was successful.

        Returns:
            bool: True if image processing was successful, False otherwise.
        """
        return self.image_processed_successfully
