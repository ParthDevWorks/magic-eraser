import pytest


def local_coverage():
    pytest_args = ["--cov=magic_eraser", "--cov-report=lcov:lcov.info", "tests/"]
    pytest.main(pytest_args)


if __name__ == "__main__":
    local_coverage()
