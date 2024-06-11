from src.common import MCastResponse
from src.common.structures import Pose
from pydantic import Field


class GetTargetPosesResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_target_poses"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    target_poses: list[Pose]
