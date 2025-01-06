from typing import Literal, List

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """
    A class to manage and access configuration settings for the Magic Eraser project.

    Configurations:
        segmentation_model (bool): Boolean indicating whether to use segmentation model. Defaults to False
        segmentation_model_id (str): The ID of the segmentation model. Defaults to "mask_rcnn".
        inpainting_model (bool): Boolean indicating whether to use inpainting model. Defaults to False
        inpainting_model_id (str): The ID of the inpainting model. Defaults to "lama_onnx".
        mode (str): The mode for which the configuration is being used. Defaults to "erase".
        target (List): A list of targets to apply the mode to. Defaults to ["person"].

    """

    model_config = ConfigDict(strict=True, extra="forbid")

    segmentation_model: bool = False
    segmentation_model_id: Literal["mask_rcnn"] = "mask_rcnn"
    inpainting_model: bool = False
    inpainting_model_id: Literal["lama_onnx", "runwayml"] = "lama_onnx"
    mode: Literal["erase", "color_splash", "remove_background"] = "erase"
    target: List = ["person"]
