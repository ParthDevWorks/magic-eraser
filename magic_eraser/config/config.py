from typing import Union

from magic_eraser.segmentation.base import SegmentationModel
from magic_eraser.inpainting.base import InpaintingModel

DEFAULT_SEGMENTATION_MODEL_ID = "mask_rcnn"
DEFAULT_INPAINTING_MODEL_ID = "lama_onnx"
VALID_MODES = ["erase_humans"]


class Config:
    """
    A class to manage and access configuration settings for the Magic Eraser project.

    Configurations:
        segmentation_model (bool): Boolean indicating whether to use segmentation model. Defaults to False
        segmentation_model_id (str): The ID of the segmentation model. Defaults to "mask_rcnn".
        inpainting_model (bool): Boolean indicating whether to use inpainting model. Defaults to False
        inpainting_model_id (str): The ID of the inpainting model. Defaults to "lama_onnx".
        mode (str): The mode for which the configuration is being used. Valid modes are VALID_MODES

    """

    def __init__(self, config: dict, mode: str):
        """
        Initialize the Config object with a configuration dictionary provided.

        Args:
            config (dict): The configuration dictionary to be used.
            mode (str): The mode for which the configuration is being used.

        Raises:
            ValueError: If config is not dictionary or config is None
            ValueError: If mode is not str or mode is None

        """
        if config is None or not isinstance(config, dict) or not config:
            raise ValueError(
                f"Invalid input for 'config'. Either, Expected dictionary, got {type(config)} OR Dictionary is empty."
            )

        if mode is None or not isinstance(mode, str) or len(mode.strip()) == 0:
            raise ValueError(
                f"Invalid input for 'mode'. Either Expected str, got {type(mode)} OR Mode is empty string."
            )

        if mode not in VALID_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. Valid modes are {VALID_MODES}."
            )

        self.mode = mode

        self.segmentation_model = None
        self.inpainting_model = None

        self.global_config = self.validate_config(config)

    def _validate_erase_humans(self, config_dict: dict) -> dict:
        for key, value in config_dict.items():

            if key == "segmentation_model" or key == "inpainting_model":
                if value is None:
                    config_dict[key] = True
                else:
                    if not isinstance(value, bool):
                        raise ValueError(
                            f"Invalid type for key '{key}'. Expected bool, got {type(value).__name__}."
                        )

            elif key == "segmentation_model_id" or key == "inpainting_model_id":
                if value is None:
                    if key == "segmentation_model_id":
                        config_dict[key] = DEFAULT_SEGMENTATION_MODEL_ID
                    if key == "inpainting_model_id":
                        config_dict[key] = DEFAULT_INPAINTING_MODEL_ID
                    else:
                        raise ValueError(
                            f"The Default ID for provided key '{key}' is not defined"
                        )
                else:
                    if not isinstance(value, str):
                        raise ValueError(
                            f"Invalid type for key '{key}'. Expected str, got {type(value).__name__}."
                        )

            else:
                raise ValueError(f"Unknown key '{key}' in the configuration.")

        return config_dict

    def validate_config(self, config_dict: dict) -> dict:
        """
        Raises:
            ValueError: If the configuration dictionary has invalid type of value.
        """

        if self.mode == "erase_humans":
            return self._validate_erase_humans(config_dict)
        else:
            raise ValueError(f"No implementation found for mode = '{self.mode}'.")

    def load_segmentation_model(self) -> None:
        """
        Load and initialize the segmentation model based on the configuration settings.

        This method is responsible for loading the appropriate segmentation model as specified in the global configuration.
        It checks if the segmentation model should be used, and if so, it loads the model with the given ID.

        Inorder to use the segmentation model, use function 'get_segmentation_model' to get the model object.

        Notes:
            - The method first checks if the segmentation model should be used based on the 'segmentation_model' key in the global configuration.
            - If the segmentation model should be used, it retrieves the 'segmentation_model_id' from the global configuration.
        """
        if not self.global_config["segmentation_model"]:
            return

        if self.segmentation_model is None:
            from magic_eraser.segmentation.factory import get_segmentation_model

            self.segmentation_model = get_segmentation_model(
                id=self.global_config["segmentation_model_id"], initialize=True
            )

    def get_segmentation_model(self) -> Union[SegmentationModel, None]:
        return self.segmentation_model

    def load_inpainting_model(self) -> None:
        """
        Load and initialize the inpainting model based on the configuration settings.

        This method is responsible for loading the appropriate inpainting model as specified in the global configuration.
        It checks if the inpainting model should be used, and if so, it loads the model with the given ID.

        Inorder to use the inpainting model, use function 'get_inpainting_model' to get the model object.

        Notes:
            - The method first checks if the inpainting model should be used based on the 'inpainting_model' key in the global configuration.
            - If the inpainting model should be used, it retrieves the 'inpainting_model_id' from the global configuration.
        """
        if not self.global_config["inpainting_model"]:
            return

        if self.inpainting_model is None:
            from magic_eraser.inpainting.factory import get_inpainitng_model

            self.inpainting_model = get_inpainitng_model(
                id=self.global_config["inpainting_model_id"], initialize=True
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
