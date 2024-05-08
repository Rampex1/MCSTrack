from src.common import MCastRequest
from pydantic import Field


class StartCaptureRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "start_capture"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
