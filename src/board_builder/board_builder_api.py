from .api import \
    LocateReferenceMarkersRequest, \
    CollectDataRequest, \
    BuildBoardRequest, \
    SetIntrinsicParametersRequest, \
    SetReferenceMarkerRequest, \
    StartBoardBuilderRequest, \
    StopBoardBuilderRequest
from .exceptions import BoardBuilderException
from .fileio import BoardBuilderConfiguration
from .board_builder import BoardBuilder
from .structures import \
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
            LocateReferenceMarkersRequest: self.locate_reference_markers,
            CollectDataRequest: self.collect_data,
            BuildBoardRequest: self.build_board,
            SetIntrinsicParametersRequest: self.set_intrinsic_parameters,
            StartBoardBuilderRequest: self.start_board_builder,
            StopBoardBuilderRequest: self.stop_board_builder})
        return return_value

    def locate_reference_markers(self, **kwargs) -> EmptyResponse:
        request: LocateReferenceMarkersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=LocateReferenceMarkersRequest)
        self._board_builder.locate_reference_markers(
            ids=request.ids,
            corners=request.corners)
        return EmptyResponse()

    def collect_data(self, **kwargs) -> dict:
        request: CollectDataRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=CollectDataRequest)
        corners_dict = self._board_builder.collect_data(
            ids=request.ids,
            corners=request.corners)
        return corners_dict

    def build_board(self, **kwargs) -> dict:
        request: BuildBoardRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=BuildBoardRequest)
        corners_dict = self._board_builder.build_board(
            ids=request.ids,
            corners=request.corners)
        return corners_dict

    # TODO: Implement this
    def set_intrinsic_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: SetIntrinsicParametersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetIntrinsicParametersRequest)
        self._board_builder.set_intrinsic_parameters(
            detector_label=request.detector_label,
            intrinsic_parameters=request.intrinsic_parameters)
        return EmptyResponse()

    def start_board_builder(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = BoardBuilderStatus.Build.RUNNING
        return EmptyResponse()

    def stop_board_builder(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = BoardBuilderStatus.Build.STOPPED
        return EmptyResponse()
