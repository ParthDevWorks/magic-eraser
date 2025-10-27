import os

import pytest

import magic_eraser
from magic_eraser.utils.image import load_image
from magic_eraser.models import get_model
from magic_eraser.models.segmentation.mask_rcnn.main import MaskRcnn
from magic_eraser.models.segmentation.utils.segmentation_output import (
    SegmentationOutput,
)

data_root = os.path.join(os.path.dirname(magic_eraser.__file__), "../sample_data")


@pytest.fixture(scope="module")
def image():
    return load_image(os.path.join(data_root, "segmentation", "sample_6.jpg"))


def test_incorrect_factory():
    with pytest.raises(ValueError):
        _ = get_model(model_id="random", disable_cache=True)


def test_check_initialization(maskrcnn_model_initialized):
    assert isinstance(maskrcnn_model_initialized, MaskRcnn)


def test_shutdown(image):
    model = get_model(model_id="mask_rcnn", disable_cache=True)
    model.shutdown()

    with pytest.raises(ValueError):
        model.inference(image_tensor=image)


def test_inference(maskrcnn_model_initialized, image):
    output = maskrcnn_model_initialized.inference(image_tensor=image)

    assert isinstance(output, list)
    for item in output:
        assert isinstance(item, SegmentationOutput)
