import pytest

from magic_eraser.segmentation.factory import get_segmentation_model
from magic_eraser.segmentation.models.mask_rcnn.main import MaskRcnn
from magic_eraser.segmentation.models.facebook_maskformer.main import FacebookMaskFormer


@pytest.fixture(scope="session")
def default_config_erase_humans():
    config_dict = {
        "mode": "erase",
        "target": ["person"],
    }

    return config_dict


@pytest.fixture(scope="session")
def maskrcnn_model_initialized() -> MaskRcnn:
    return get_segmentation_model(id="mask_rcnn")


@pytest.fixture(scope="session")
def maskformer_model_initialized() -> FacebookMaskFormer:
    return get_segmentation_model(id="maskformer")
