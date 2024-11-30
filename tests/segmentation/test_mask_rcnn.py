import os

import pytest
import torch

import magic_eraser
from magic_eraser.utils.image import load_image
from magic_eraser.segmentation.factory import get_segmentation_model
from magic_eraser.segmentation.models.mask_rcnn.main import MaskRcnn
from magic_eraser.segmentation.utils.segmentation_output import SegmentationOutput

data_root = os.path.join(os.path.dirname(magic_eraser.__file__), "../sample_data")


@pytest.fixture(scope="module")
def image():
    return load_image(os.path.join(data_root, "segmentation", "sample_6.jpg"))


def test_correct_factory_initialized(maskrcnn_model_initialized):
    assert maskrcnn_model_initialized.get_model_id() == "mask_rcnn"


def test_correct_factory_uninitialized(model_uninitialized):
    assert model_uninitialized.get_model_id() == "mask_rcnn"


def test_incorrect_factory():
    with pytest.raises(ValueError):
        _ = get_segmentation_model(id="random")


def test_check_initialization(maskrcnn_model_initialized):
    assert isinstance(maskrcnn_model_initialized, MaskRcnn)


def test_check_incorrect_initialization(model_uninitialized, image):
    with pytest.raises(ValueError):
        model_uninitialized.inference(image_tensor=image)


def test_shutdown(image):
    model = get_segmentation_model(id="mask_rcnn", initialize=True)
    model.shutdown()

    with pytest.raises(ValueError):
        model.inference(image_tensor=image)


def test_inference(maskrcnn_model_initialized, image):
    output = maskrcnn_model_initialized.inference(image_tensor=image)

    assert isinstance(output, SegmentationOutput)
