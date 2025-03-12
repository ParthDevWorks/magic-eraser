from typing import Union

from magic_eraser.config.config import (
    Config,
    DEFAULT_SEGMENTATION_MODEL_ID,
    DEFAULT_INPAINTING_MODEL_ID,
)
from magic_eraser.segmentation.base import SegmentationModel
from magic_eraser.inpainting.base import InpaintingModel


class ModelInitializer:
    """
    This class is responsible for initializing and managing models used in the MAGIC_ERASER project.

    It handles the loading and initialization of both segmentation and inpainting models.

    The class provides methods to load specific models, retrieve loaded models, and shut down models when necessary.

    Attributes:
        global_config (Config): Global configuration object containing model-related settings.
    """

    def __init__(self, global_config: Config) -> None:
        self.segmentation_model = None
        self.inpainting_model = None
        self.global_config = global_config.model_dump()

    def load_segmentation_model(self) -> None:
        """
        Load and initialize the segmentation model.

        This method is responsible for loading the segmentation model.

        Inorder to use the segmentation model, use function 'get_segmentation_model' to get the model object.
        """

        if self.segmentation_model is None:
            from magic_eraser.segmentation.factory import get_segmentation_model

            self.segmentation_model = get_segmentation_model(
                id=DEFAULT_SEGMENTATION_MODEL_ID, initialize=True
            )

    def get_segmentation_model(self) -> Union[SegmentationModel, None]:
        return self.segmentation_model

    def load_inpainting_model(self) -> None:
        """
        Load and initialize the inpainting model.

        This method is responsible for loading the inpainting model.

        Inorder to use the inpainting model, use function 'get_inpainting_model' to get the model object.
        """

        if self.inpainting_model is None:
            from magic_eraser.inpainting.factory import get_inpainitng_model

            self.inpainting_model = get_inpainitng_model(
                id=DEFAULT_INPAINTING_MODEL_ID, initialize=True
            )

    def get_inpainting_model(self) -> Union[InpaintingModel, None]:
        return self.inpainting_model

    def load_models(self) -> None:
        self.load_segmentation_model()
        self.load_inpainting_model()

    def shutdown_models(self) -> None:
        if self.segmentation_model is not None:
            self.segmentation_model.shutdown()
            self.segmentation_model = None

        if self.inpainting_model is not None:
            self.inpainting_model.shutdown()
            self.inpainting_model = None
