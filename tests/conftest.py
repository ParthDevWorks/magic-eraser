import pytest

from magic_eraser.models import get_model
from magic_eraser.models.segmentation.mask_rcnn.main import MaskRcnn
from magic_eraser.models.segmentation.facebook_maskformer.main import (
    FacebookMaskFormer,
)


@pytest.fixture(scope="session")
def default_config_erase_humans():
    config_dict = {
        "mode": "erase",
        "target": ["person"],
    }

    return config_dict


@pytest.fixture(scope="session")
def maskrcnn_model_initialized() -> MaskRcnn:
    return get_model(model_id="mask_rcnn", disable_cache=True)


@pytest.fixture(scope="session")
def maskformer_model_initialized() -> FacebookMaskFormer:
    return get_model(model_id="maskformer", disable_cache=True)
