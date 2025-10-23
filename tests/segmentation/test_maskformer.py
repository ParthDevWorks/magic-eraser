import os

import pytest

import magic_eraser
from magic_eraser.utils.image import load_image
from magic_eraser.segmentation.factory import get_segmentation_model
from magic_eraser.segmentation.models.facebook_maskformer.main import FacebookMaskFormer
from magic_eraser.segmentation.utils.segmentation_output import SegmentationOutput

data_root = os.path.join(os.path.dirname(magic_eraser.__file__), "../sample_data")


@pytest.fixture(scope="module")
def image():
    return load_image(os.path.join(data_root, "segmentation", "sample_1.jpg"))


def test_incorrect_factory():
    with pytest.raises(ValueError):
        _ = get_segmentation_model(id="random")


def test_check_initialization(maskformer_model_initialized):
    assert isinstance(maskformer_model_initialized, FacebookMaskFormer)


def test_shutdown(image):
    model = get_segmentation_model(id="maskformer")
    model.shutdown()

    with pytest.raises(ValueError):
        model.inference(image_tensor=image)


def test_inference(maskformer_model_initialized, image):
    output = maskformer_model_initialized.inference(image_tensor=image)

    assert isinstance(output, list)
    for item in output:
        assert isinstance(item, SegmentationOutput)
