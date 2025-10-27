from magic_eraser.models.base import BaseModel

SEGMENTATION_MODELS = ["mask_rcnn", "maskformer"]
INPAINTING_MODELS = ["lama_onnx"]  # "runwayml" Commented until further time
OCR_MODELS = ["doctr"]

VALID_MODEL_IDS = SEGMENTATION_MODELS + INPAINTING_MODELS + OCR_MODELS

_model_cache = {}


def _model_factory(id) -> BaseModel:
    """This method is the entry point to load any model.

    Args:
        id (str): Model Id
    """

    # Segmentation Model Factory
    if id == "mask_rcnn":
        from magic_eraser.models.segmentation.mask_rcnn.main import MaskRcnn

        model = MaskRcnn()
    elif id == "maskformer":
        from magic_eraser.models.segmentation.facebook_maskformer.main import (
            FacebookMaskFormer,
        )

        model = FacebookMaskFormer()

    # OCR Model Factory
    elif id == "doctr":
        from magic_eraser.models.ocr.doctr.main import Doctr

        model = Doctr()

    # InPainting Model Factory
    elif id == "lama_onnx":
        from magic_eraser.models.inpainting.lama.main import LamaOnnx

        model = LamaOnnx()

    # Not to be used
    # elif id == "runwayml":
    #     from magic_eraser.models.inpainting.runwayml.main import RunWayML

    #     model = RunWayML()

    # Model Factory Not Found
    else:
        raise ValueError(
            f"Supported Model ids are {VALID_MODEL_IDS} but user provided id as {id}"
        )
    assert model.get_model_id() == id

    return model


def get_model(model_id: str, disable_cache: bool = False) -> BaseModel:
    if disable_cache:
        return _model_factory(model_id)

    try:
        return _model_cache[model_id]
    except KeyError:
        initialize_all_models()


def initialize_all_models() -> None:
    for model_id in VALID_MODEL_IDS:
        _model_cache[model_id] = _model_factory(model_id)
