import pytest
from magic_eraser.utils.book_keeping import (
    Logs,
    SuccessLogs,
    ErrorLogs,
    FatalProcessingError,
)


@pytest.mark.parametrize(
    "in_path, mode, message, expected_logs",
    [
        (
            "a/b/c",
            "eraser",
            "Image Found",
            {"input_path": "a/b/c", "mode": "eraser", "message": "Image Found"},
        ),
        (None, None, None, {"input_path": None, "mode": None, "message": None}),
        (
            "input_path1",
            None,
            None,
            {"input_path": "input_path1", "mode": None, "message": None},
        ),
        (None, "mode1", None, {"input_path": None, "mode": "mode1", "message": None}),
        (
            None,
            None,
            "message1",
            {"input_path": None, "mode": None, "message": "message1"},
        ),
    ],
)
def test_logs(in_path, mode, message, expected_logs):
    actual_logs = Logs(input_path=in_path, mode=mode, message=message)
    assert actual_logs.as_dict() == expected_logs


@pytest.mark.parametrize(
    "in_path, mode, message, out_path, infer_time, expected_logs",
    [
        (
            "a/b/c",
            "eraser",
            "Image Found",
            "output/a",
            12.34,
            {
                "input_path": "a/b/c",
                "mode": "eraser",
                "message": "Image Found",
                "output_path": "output/a",
                "inference_time_seconds": 12.34,
            },
        ),
        (
            None,
            None,
            None,
            None,
            None,
            {
                "input_path": None,
                "mode": None,
                "message": None,
                "output_path": None,
                "inference_time_seconds": None,
            },
        ),
    ],
)
def test_success_logs(in_path, mode, message, out_path, infer_time, expected_logs):
    actual_logs = SuccessLogs(
        input_path=in_path,
        mode=mode,
        message=message,
        output_path=out_path,
        inference_time_seconds=infer_time,
    )
    assert actual_logs.as_dict() == expected_logs


@pytest.mark.parametrize(
    "in_path, mode, message, traceback, expected_logs",
    [
        (
            "a/b/c",
            "eraser",
            "Cant process",
            "Invalid File",
            {
                "input_path": "a/b/c",
                "mode": "eraser",
                "message": "Cant process",
                "traceback": "Invalid File",
            },
        ),
        (
            None,
            None,
            None,
            None,
            {
                "input_path": None,
                "mode": None,
                "message": None,
                "traceback": None,
            },
        ),
    ],
)
def test_error_logs(in_path, mode, message, traceback, expected_logs):
    actual_logs = ErrorLogs(
        input_path=in_path,
        mode=mode,
        message=message,
        traceback=traceback,
    )
    assert actual_logs.as_dict() == expected_logs


@pytest.mark.parametrize(
    "message, traceback, expected_logs",
    [
        (
            "Error",
            "Division by zero",
            {"message": "Error", "traceback": "Division by zero"},
        )
    ],
)
def test_fatal_processing_logs(message, traceback, expected_logs):
    actual_logs = FatalProcessingError(message=message, traceback=traceback)
    assert actual_logs.as_dict() == expected_logs
