from src.common import MCastRequest
from pydantic import Field


class SetReferenceMarkerRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "set_reference_marker"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    marker_id: int = Field()
    marker_diameter: float = Field()
