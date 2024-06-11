from typing import Final
from enum import IntEnum


class BoardBuilderStatus:

    class Build(IntEnum):
        STOPPED: Final[int] = 0
        RUNNING: Final[int] = 1
        FAILURE: Final[int] = 2

    solve_status: Build
    solve_errors: list[str]

    def __init__(self):
        self.solve_status = BoardBuilderStatus.Build.STOPPED
        self.solve_errors = list()

    def in_runnable_state(self):
        return self.solve_status == BoardBuilderStatus.Build.RUNNING
