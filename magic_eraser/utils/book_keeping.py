from typing import Union


class Logs:
    __slots__ = ["input_path", "mode", "message"]

    def __init__(
        self,
        input_path: Union[str, None] = None,
        mode: Union[str, None] = None,
        message: Union[str, None] = None,
    ):
        self.input_path = input_path
        self.mode = mode
        self.message = message

    def as_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __str__(self):
        return self.as_dict().__str__()


class ErrorLogs(Logs):
    __slots__ = Logs.__slots__ + ["traceback"]

    def __init__(
        self,
        input_path: Union[str, None] = None,
        mode: Union[str, None] = None,
        message: Union[str, None] = None,
        traceback: Union[str, None] = None,
    ):
        super().__init__(input_path=input_path, mode=mode, message=message)
        self.traceback = traceback


class SuccessLogs(Logs):
    __slots__ = Logs.__slots__ + ["output_path", "inference_time_seconds"]

    def __init__(
        self,
        input_path: Union[str, None] = None,
        output_path: Union[str, None] = None,
        mode: Union[str, None] = None,
        message: Union[str, None] = None,
        inference_time_seconds: Union[float, None] = None,
    ):
        super().__init__(input_path=input_path, mode=mode, message=message)
        self.output_path = output_path
        self.inference_time_seconds = inference_time_seconds


class FatalProcessingError(Logs):
    __slots__ = ["message", "traceback"]

    def __init__(
        self,
        message: str = "Fatal Processing Error",
        traceback: str = None,
    ):
        self.message = message
        self.traceback = traceback


if __name__ == "__main__":
    dummy_error_message = ErrorLogs(
        input_path="/path/to/test.jpg",
        mode="erase_humans",
        message="Import Error",
        traceback="ModuleNotFoundError: No module named 'magic_eraser'",
    )

    print(dummy_error_message)
    print(type(dummy_error_message))
    print("***" * 20)

    print(dummy_error_message.as_dict())
    print(type(dummy_error_message.as_dict()))
    print("***" * 20)

    print(dummy_error_message.__str__())
    print(type(dummy_error_message.__str__()))
