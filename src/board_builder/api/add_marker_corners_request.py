from src.common import MCastRequest
from src.common.structures import MarkerSnapshot
from pydantic import Field


class AddMarkerCornersRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "add_marker_corners"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    detected_marker_snapshots: list[MarkerSnapshot] | None = Field()
    rejected_marker_snapshots: list[MarkerSnapshot] | None = Field()
    detector_label: str = Field()
    detector_timestamp_utc_iso8601: str = Field()
