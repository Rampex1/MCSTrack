from typing import Final
from enum import IntEnum


class CaptureStatus:

    class Status(IntEnum):
        STOPPED: Final[int] = 0
        RUNNING: Final[int] = 1
        FAILURE: Final[int] = 2

    status: Status
    errors: list[str]

    def __init__(self):
        self.status = CaptureStatus.Status.STOPPED
        self.errors = list()

    def in_runnable_state(self):
        return self.status == CaptureStatus.Status.RUNNING
