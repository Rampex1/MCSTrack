from src.common import MCastResponse
from pydantic import Field


class AddTargetMarkerResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "add_marker_corners"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    target_id: str = Field()
