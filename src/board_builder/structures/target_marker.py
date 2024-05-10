from .target_base import Target
from pydantic import Field


class TargetMarker(Target):
    marker_id: int = Field()
    marker_size: float = Field()
