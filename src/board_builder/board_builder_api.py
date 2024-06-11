from .api import \
    AddMarkerCornersRequest, \
    AddTargetMarkerRequest, \
    AddTargetMarkerResponse, \
    GetDetectorPosesRequest, \
    GetDetectorPosesResponse, \
    GetTargetPosesRequest, \
    GetTargetPosesResponse, \
    SetIntrinsicParametersRequest, \
    SetReferenceMarkerRequest, \
    StartBoardBuilderRequest, \
    StopBoardBuilderRequest
from .exceptions import BoardBuilderException
from .fileio import BoardBuilderConfiguration
from .board_builder import BoardBuilder
from .structures import \
    MarkerCorners, \
    TargetMarker
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCastComponent, \
    MCastRequest, \
    MCastResponse
from src.common.structures import \
    Pose, \
    BoardBuilderStatus
import datetime
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def dummy_function(_: dict) -> MCastResponse:
    return EmptyResponse()


class BoardBuilderAPI(MCastComponent):
    """
    API-friendly layer overtop of a BoardBuilder
    """
    _status: BoardBuilderStatus
    _board_builder: BoardBuilder

    def __init__(
        self,
        configuration: BoardBuilderConfiguration,
        board_builder: BoardBuilder
    ):
        super().__init__(
            status_source_label=configuration.serial_identifier,
            send_status_messages_to_logger=True)
        self._board_builder = board_builder
        self._status = BoardBuilderStatus()

    async def internal_update(self):
        if self._status.solve_status == BoardBuilderStatus.Build.RUNNING:
            self._board_builder.update()

    def supported_request_types(self) -> dict[type[MCastRequest], Callable[[dict], MCastResponse]]:
        return_value: dict[type[MCastRequest], Callable[[dict], MCastResponse]] = super().supported_request_types()
        return_value.update({
            AddMarkerCornersRequest: self.add_marker_corners,
            AddTargetMarkerRequest: self.add_target_marker,
            GetDetectorPosesRequest: self.get_detector_poses,
            GetTargetPosesRequest: self.get_target_poses,
            SetIntrinsicParametersRequest: self.set_intrinsic_parameters,
            SetReferenceMarkerRequest: self.set_reference_marker,
            StartBoardBuilderRequest: self.start_board_builder,
            StopBoardBuilderRequest: self.stop_board_builder})
        return return_value

    def add_marker_corners(self, **kwargs) -> EmptyResponse:
        request: AddMarkerCornersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=AddMarkerCornersRequest)
        detector_timestamp_utc: datetime.datetime = datetime.datetime.fromisoformat(
            request.detector_timestamp_utc_iso8601)  # TODO: ErrorResponse if formatted incorrectly?
        detected_corners: list[MarkerCorners] = [
            MarkerCorners(
                detector_label=request.detector_label,
                marker_id=int(detected_marker_snapshot.label),
                points=[
                    [detected_marker_snapshot.corner_image_points[i].x_px,
                     detected_marker_snapshot.corner_image_points[i].y_px]
                    for i in range(0, 4)],
                timestamp=detector_timestamp_utc)
            for detected_marker_snapshot in request.detected_marker_snapshots]
        self._board_builder.add_marker_corners(detected_corners=detected_corners)
        return EmptyResponse()

    def add_target_marker(self, **kwargs) -> AddTargetMarkerResponse | ErrorResponse:
        request: AddTargetMarkerRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=AddTargetMarkerRequest)
        try:
            target_id: str = self._board_builder.add_target_marker(
                marker_id=request.marker_id,
                marker_diameter=request.marker_diameter)
        except BoardBuilderException as e:
            return ErrorResponse(
                message=e.message)
        return AddTargetMarkerResponse(
            target_id=target_id)

    def get_poses(self, **_kwargs) -> GetPosesResponse:
        detector_poses: list[Pose]
        target_poses: list[Pose]
        detector_poses, target_poses = self._board_builder.get_poses()
        return GetPosesResponse(
            detector_poses=detector_poses,
            target_poses=target_poses)

    def set_intrinsic_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: SetIntrinsicParametersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetIntrinsicParametersRequest)
        self._board_builder.set_intrinsic_parameters(
            detector_label=request.detector_label,
            intrinsic_parameters=request.intrinsic_parameters)
        return EmptyResponse()

    def set_reference_marker(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: SetReferenceMarkerRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetReferenceMarkerRequest)
        try:
            self._board_builder.set_reference_target(
                TargetMarker(
                    marker_id=request.marker_id,
                    marker_size=request.marker_diameter))
        except BoardBuilderException as e:
            return ErrorResponse(
                message=e.message)
        return EmptyResponse()

    def start_board_builder(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = BoardBuilderStatus.Build.RUNNING
        return EmptyResponse()

    def stop_board_builder(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = BoardBuilderStatus.Build.STOPPED
        return EmptyResponse()
