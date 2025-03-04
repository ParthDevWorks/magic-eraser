import pytest


def local_coverage():
    pytest_args = ["--cov=magic_eraser", "--cov-report=lcov:lcov.info", "tests/"]
    pytest.main(pytest_args)


def terminal_coverage():
    pytest_args = ["--cov=magic_eraser", "--cov-report=term", "tests/"]
    pytest.main(pytest_args)
