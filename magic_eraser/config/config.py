from typing import Literal, List

from pydantic import BaseModel, ConfigDict

DEFAULT_SEGMENTATION_MODEL_ID = "mask_rcnn"
DEFAULT_INPAINTING_MODEL_ID = "lama_onnx"


class Config(BaseModel):
    """
    A class to manage and access configuration settings for the Magic Eraser project.

    Configurations:
        mode (str): The mode for which the configuration is being used. Defaults to "erase".
        target (List): A list of targets to apply the mode to. Defaults to ["person"].

    """

    model_config = ConfigDict(strict=True, extra="forbid")

    mode: Literal["erase", "color_splash", "remove_background"] = "erase"
    target: List = ["person"]
