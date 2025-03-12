import os

import pytest
import torch

import magic_eraser
from magic_eraser.inpainting.factory import get_inpainitng_model
from magic_eraser.inpainting.models.lama.main import LamOnnx
from magic_eraser.utils.image import load_image

data_root = os.path.join(os.path.dirname(magic_eraser.__file__), "../sample_data")


@pytest.fixture(scope="module")
def model_initialized():
    return get_inpainitng_model(id="lama_onnx")


@pytest.fixture(scope="module")
def image():
    return load_image(os.path.join(data_root, "inpainting", "image_1.jpg"))


@pytest.fixture(scope="module")
def mask():
    return load_image(os.path.join(data_root, "inpainting", "mask_1.png"))


def test_correct_factory_initialized(model_initialized):
    assert model_initialized.get_model_id() == "lama_onnx"


def test_incorrect_factory():
    with pytest.raises(ValueError):
        _ = get_inpainitng_model(id="random")


def test_check_initialization(model_initialized):
    assert isinstance(model_initialized, LamOnnx)


def test_shutdown(image, mask):
    model = get_inpainitng_model(id="lama_onnx")
    model.shutdown()

    with pytest.raises(ValueError):
        model.inference(image=image, mask=mask)


def test_inference(model_initialized, image, mask):
    output = model_initialized.inference(image=image, mask=mask)

    assert isinstance(output, torch.Tensor)


def test_image_shape(model_initialized, image, mask):
    output = model_initialized.inference(image=image, mask=mask)
    assert output.shape == image.shape
