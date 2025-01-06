import pytest

from magic_eraser.segmentation.factory import get_segmentation_model


@pytest.fixture(scope="session")
def default_config_erase_humans():
    config_dict = {
        "segmentation_model": True,
        "segmentation_model_id": "mask_rcnn",
        "inpainting_model": True,
        "inpainting_model_id": "lama_onnx",
        "mode": "erase",
        "target": ["person"],
    }

    return config_dict


@pytest.fixture(scope="session")
def maskrcnn_model_initialized():
    return get_segmentation_model(id="mask_rcnn", initialize=True)


@pytest.fixture(scope="session")
def maskrcnn_model_uninitialized():
    return get_segmentation_model(id="mask_rcnn", initialize=False)
