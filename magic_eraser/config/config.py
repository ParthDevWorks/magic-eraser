from typing import Literal

from pydantic import BaseModel, ConfigDict

DEFAULT_SEGMENTATION_MODEL_ID = "maskformer"
DEFAULT_INPAINTING_MODEL_ID = "lama_onnx"
DEFAULT_OCR_MODEL_ID = "doctr"


class Config(BaseModel):
    """
    A class to manage and access configuration settings for the Magic Eraser project.

    Configurations:
        mode (str): The mode for which the configuration is being used. Defaults to "erase".
        target (List): A list of targets to apply the mode to. Defaults to ["person"].

    """

    model_config = ConfigDict(strict=True, extra="forbid")

    mode: Literal[
        "erase", "color_splash", "remove_background", "remove_text", "fall_color"
    ] = "erase"
    target: list = ["person"]
