from .pose import Pose
import datetime
from pydantic import BaseModel, Field


class PoseSolverFrame(BaseModel):
    detector_poses: list[Pose] | None = Field()
    target_poses: list[Pose] | None = Field()
    timestamp_utc_iso8601: str = Field()

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)
