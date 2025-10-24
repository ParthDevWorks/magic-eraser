import torch

from magic_eraser.config.config import Config
from magic_eraser.model_initialization.initialize import ModelInitializer
from magic_eraser.segmentation.base import SegmentationModel
from magic_eraser.inpainting.base import InpaintingModel
from magic_eraser.ocr.base import OCRModel
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
        model_initializer (ModelInitializer): An instance of ModelInitializer for loading models and their config.
    """

    def __init__(self, model_initializer: ModelInitializer) -> None:
        self.model_initializer = model_initializer
        self.global_config: Config = model_initializer.global_config

    def analyze_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Analyzes the input image based on the configured mode and processes it accordingly.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Processed image tensor.
        """

        self.image_processed_successfully = False

        if self.global_config["mode"] == "erase":
            output_image_tensor = erase(
                image_tensor,
                self.segmentation_model,
                self.inpainting_model,
                self.global_config["target"],
            )

        elif self.global_config["mode"] == "color_splash":
            output_image_tensor = color_splash(
                image_tensor, self.segmentation_model, self.global_config["target"]
            )

        elif self.global_config["mode"] == "remove_background":
            output_image_tensor = remove_background(
                image_tensor, self.segmentation_model, self.global_config["target"]
            )

        elif self.global_config["mode"] == "remove_text":
            output_image_tensor = remove_text(image_tensor, self.ocr_model)

        elif self.global_config["mode"] == "fall_color":
            output_image_tensor = fall_color(image_tensor, self.segmentation_model)

        self.image_processed_successfully = True
        return output_image_tensor

    def load_models(self) -> None:
        """
        Loads the required models for image processing.
        """
        self.model_initializer.load_models()

    @property
    def segmentation_model(self) -> SegmentationModel | None:
        """
        Gets the SegmentationModel instance.

        Returns:
            SegmentationModel | None: Instance of SegmentationModel or None if not initialized.
        """
        return self.model_initializer.get_segmentation_model()

    @property
    def inpainting_model(self) -> InpaintingModel | None:
        """
        Gets the InpaintingModel instance.

        Returns:
            InpaintingModel | None: Instance of InpaintingModel or None if not initialized.
        """
        return self.model_initializer.get_inpainting_model()

    @property
    def ocr_model(self) -> OCRModel | None:
        """
        Gets the OCRModel instance.

        Returns:
            OCRModel | None: Instance of OCRModel or None if not initialized.
        """
        return self.model_initializer.get_ocr_model()

    @property
    def is_image_processing_successfull(self) -> bool:
        """
        Checks if the image processing was successful.

        Returns:
            bool: True if image processing was successful, False otherwise.
        """
        return self.image_processed_successfully
