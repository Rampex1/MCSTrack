from .marker_snapshot import MarkerSnapshot
import datetime
from pydantic import BaseModel, Field


class DetectorFrame(BaseModel):
    detected_marker_snapshots: list[MarkerSnapshot] | None = Field()
    rejected_marker_snapshots: list[MarkerSnapshot] | None = Field()
    timestamp_utc_iso8601: str = Field()

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)
