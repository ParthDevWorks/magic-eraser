import os
import tempfile

import pytest

from magic_eraser.core import eraser
from magic_eraser.models import initialize_all_models

data_root = str(os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data"))


@pytest.mark.parametrize(
    "mode", ["erase", "color_splash", "remove_background", "fall_color"]
)
def test_entire_code(default_config_erase_humans, mode):
    default_config_erase_humans["mode"] = mode
    initialize_all_models()

    with tempfile.TemporaryDirectory() as output_temp_dir:
        success_cnt, error_cnt = eraser(
            config_dict=default_config_erase_humans,
            input_path=os.path.join(data_root, "segmentation", "sample_1.jpg"),
            output_folder_dir=output_temp_dir,
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
