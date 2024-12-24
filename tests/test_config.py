from magic_eraser.config.config import Config


def test_valid_config(default_config_erase_humans):
    config = Config(**default_config_erase_humans)
    assert config.model_dump() == default_config_erase_humans
