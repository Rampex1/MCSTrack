import abc
import datetime
from src.common import \
    EmptyResponse, \
    ErrorResponse
from src.common.structures import MarkerCornerImagePoint
from src.common.structures import MarkerStatus
from src.detector.api import \
    GetDetectionParametersResponse, \
    GetMarkerSnapshotsResponse
from typing import Any

class AbstractMarkerInterface(abc.ABC):
    marker_status: MarkerStatus  # internal bookkeeping
    marker_timestamp_utc: datetime.datetime

    def set_detection_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        pass

    def get_detection_parameters(self, **_kwargs) -> GetDetectionParametersResponse | ErrorResponse:
        pass

    def get_marker_snapshots(self, **kwargs) -> GetMarkerSnapshotsResponse:
        pass

    def internal_update_marker_corners(self, marker_status) -> list[MarkerCornerImagePoint]:
        pass
