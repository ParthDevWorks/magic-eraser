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
