from src.common import MCastRequest
from pydantic import Field


class GetTargetPosesRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_target_poses"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
