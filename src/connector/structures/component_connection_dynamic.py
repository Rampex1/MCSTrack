import datetime
from typing import Final, Literal
from websockets import WebSocketClientProtocol


class ComponentConnectionDynamic:

    ATTEMPT_COUNT_MAXIMUM: Final[int] = 3
    ATTEMPT_TIME_GAP_SECONDS: Final[float] = 5.0

    status: Literal["disconnecting", "disconnected", "connecting", "connected", "aborted"]
    socket: WebSocketClientProtocol | None
    attempt_count: int
    next_attempt_timestamp_utc: datetime.datetime

    def __init__(self):
        self.status = "disconnected"
        self.socket = None
        self.attempt_count = 0
        self.next_attempt_timestamp_utc = datetime.datetime.min
