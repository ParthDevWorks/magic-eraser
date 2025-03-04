import os
import pytest
import tempfile
from unittest.mock import patch

from magic_eraser.core import eraser
from magic_eraser.utils.image import load_image
from magic_eraser.config.config import Config
from magic_eraser.model_initialization.initialize import ModelInitializer
from magic_eraser.pipeline import Pipeline

data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


@pytest.fixture(scope="module")
def image():
    return load_image(os.path.join(data_root, "segmentation", "sample_6.jpg"))


@pytest.mark.parametrize("mode", ["erase", "color_splash", "remove_background"])
def test_entire_pipeline(default_config_erase_humans, mode, image):
    default_config_erase_humans["mode"] = mode
    config = Config(**default_config_erase_humans)

    model_initializer = ModelInitializer(global_config=config)
    pipeline = Pipeline(model_initializer=model_initializer)
    pipeline.load_models()

    output = pipeline.analyze_image(image)

    assert output is not None
    assert output.shape == image.shape
    assert output.dtype == image.dtype


@pytest.mark.parametrize("mode", ["erase", "color_splash", "remove_background"])
@pytest.mark.parametrize("num_workers", [0, 1])
def test_entire_code(default_config_erase_humans, mode, num_workers):
    default_config_erase_humans["mode"] = mode

    with tempfile.TemporaryDirectory() as output_temp_dir:

        with patch("os.path.isdir", autospec=True) as mocked_isdir:
            mocked_isdir.return_value = True

            success_cnt, error_cnt = eraser(
                config_dict=default_config_erase_humans,
                input_folder_path=os.path.join(
                    data_root,
                    "segmentation",
                ),
                output_folder_path=output_temp_dir,
                num_workers=num_workers,
            )

            expected_output_format = ".PNG"
            assert error_cnt == 0
            assert success_cnt > 0

            actual_files = [
                file
                for file in os.listdir(output_temp_dir)
                if file.endswith(expected_output_format)
            ]

            assert len(actual_files) == success_cnt
