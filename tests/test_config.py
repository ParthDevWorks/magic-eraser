import pytest

from magic_eraser.config.config import Config


def test_invalid_config_type():
    with pytest.raises(ValueError):
        _ = Config(config=123, mode="erase_humans")


def test_invalid_config_none():
    with pytest.raises(ValueError):
        _ = Config(config=None, mode="erase_humans")


def test_invalid_config_emptydictionary():
    with pytest.raises(ValueError):
        _ = Config(config={}, mode="erase_humans")


def test_invalid_mode_type():
    with pytest.raises(ValueError):
        _ = Config(config={"abc": "a"}, mode=["watevr"])


def test_invalid_mode_none():
    with pytest.raises(ValueError):
        _ = Config(config={"abc": "a"}, mode=None)


def test_invalid_mode_none():
    with pytest.raises(ValueError):
        _ = Config(config={"abc": "a"}, mode=" ")


def test_valid_config(default_config_erase_humans, default_mode_erase_humans):
    config = Config(config=default_config_erase_humans, mode=default_mode_erase_humans)
    assert config.global_config == default_config_erase_humans
