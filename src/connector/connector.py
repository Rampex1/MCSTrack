from .exceptions import ResponseSeriesNotExpected
from src.common import \
    DequeueStatusMessagesRequest, \
    DequeueStatusMessagesResponse, \
    EmptyResponse, \
    ErrorResponse, \
    MCastComponent, \
    MCastRequest, \
    MCastRequestSeries, \
    MCastResponse, \
    MCastResponseSeries, \
    mcast_websocket_send_recv
from src.common.structures import \
    COMPONENT_ROLE_LABEL_CALIBRATOR, \
    COMPONENT_ROLE_LABEL_DETECTOR, \
    COMPONENT_ROLE_LABEL_POSE_SOLVER, \
    ComponentConnectionStatic
from src.connector.structures import \
    BaseComponentConnectionDynamic, \
    CalibratorComponentConnectionDynamic, \
    ConnectionTableRow, \
    DetectorComponentConnectionDynamic, \
    PoseSolverComponentConnectionDynamic
from src.calibrator.api import \
    AddCalibrationImageResponse, \
    CalibrateResponse, \
    GetCalibrationImageResponse, \
    GetCalibrationResultResponse, \
    ListCalibrationDetectorResolutionsResponse, \
    ListCalibrationImageMetadataResponse, \
    ListCalibrationResultMetadataResponse
from src.detector.api import \
    GetCaptureDeviceResponse, \
    GetCapturePropertiesResponse, \
    GetCaptureImageResponse, \
    GetDetectionParametersResponse, \
    GetMarkerSnapshotsResponse
from src.pose_solver.api import \
    AddTargetMarkerResponse, \
    GetPosesResponse
import datetime
import logging
from typing import Callable, Tuple
import uuid
from websockets import \
    connect

logger = logging.getLogger(__name__)


SUPPORTED_RESPONSE_TYPES: list[type[MCastResponse]] = [
    AddCalibrationImageResponse,
    AddTargetMarkerResponse,
    CalibrateResponse,
    DequeueStatusMessagesResponse,
    EmptyResponse,
    ErrorResponse,
    GetCalibrationImageResponse,
    GetCalibrationResultResponse,
    GetCaptureDeviceResponse,
    GetCaptureImageResponse,
    GetCapturePropertiesResponse,
    GetDetectionParametersResponse,
    GetMarkerSnapshotsResponse,
    GetPosesResponse,
    ListCalibrationDetectorResolutionsResponse,
    ListCalibrationImageMetadataResponse,
    ListCalibrationResultMetadataResponse]


class Connector(MCastComponent):

    class Connection:
        def __init__(
            self,
            static: ComponentConnectionStatic,
            dynamic: BaseComponentConnectionDynamic
        ):
            self.static = static
            self.dynamic = dynamic

        static: ComponentConnectionStatic
        dynamic: BaseComponentConnectionDynamic

    _serial_identifier: str
    _connections: dict[str, Connection]

    _request_series_by_label: dict[str, list[Tuple[MCastRequestSeries, uuid.UUID]]]

    # None indicates that no response has been received yet.
    _response_series_by_id: dict[uuid.UUID, MCastResponseSeries | None]

    def __init__(
        self,
        serial_identifier: str,
        send_status_messages_to_logger: bool = False
    ):
        super().__init__(
            status_source_label=serial_identifier,
            send_status_messages_to_logger=send_status_messages_to_logger)
        self._serial_identifier = serial_identifier
        self._connections = dict()
        self._request_series_by_label = dict()
        self._response_series_by_id = dict()

    def add_connection(
        self,
        connection_static: ComponentConnectionStatic
    ) -> None:
        label = connection_static.label
        if label in self._connections:
            raise RuntimeError(f"Connection associated with {label} already exists.")
        connection_dynamic: BaseComponentConnectionDynamic
        if connection_static.role == COMPONENT_ROLE_LABEL_CALIBRATOR:
            connection_dynamic = CalibratorComponentConnectionDynamic()
        elif connection_static.role == COMPONENT_ROLE_LABEL_DETECTOR:
            connection_dynamic = DetectorComponentConnectionDynamic()
        elif connection_static.role == COMPONENT_ROLE_LABEL_POSE_SOLVER:
            connection_dynamic = PoseSolverComponentConnectionDynamic()
        else:
            raise NotImplementedError(f"Handling for {connection_static.role} not implemented")
        self._connections[label] = Connector.Connection(
            static=connection_static,
            dynamic=connection_dynamic)

    def begin_connecting(self, label: str) -> None:
        if label not in self._connections:
            message: str = f"label {label} is not in list. Returning."
            self.add_status_message(severity="error", message=message)
            return
        if self._connections[label].dynamic.status == "connected":
            message: str = f"label {label} is already connected. Returning."
            self.add_status_message(severity="warning", message=message)
            return
        self._connections[label].dynamic.status = "connecting"
        self._connections[label].dynamic.attempt_count = 0

    def begin_disconnecting(self, label: str) -> None:
        if label not in self._connections:
            message: str = f"label {label} is not in list. Returning."
            self.add_status_message(severity="error", message=message)
            return
        self._connections[label].dynamic.status = "disconnecting"
        self._connections[label].dynamic.attempt_count = 0
        self._connections[label].dynamic.socket = None

    def contains_connection_label(self, label: str) -> bool:
        return label in self._connections

    def get_connection_table_rows(self) -> list[ConnectionTableRow]:
        return_value: list[ConnectionTableRow] = list()
        for connection in self._connections.values():
            return_value.append(ConnectionTableRow(
                label=connection.static.label,
                role=connection.static.role,
                ip_address=str(connection.static.ip_address),
                port=int(connection.static.port),
                status=connection.dynamic.status))
        return return_value

    def get_connected_detector_labels(self) -> list[str]:
        return self.get_connected_role_labels(role=COMPONENT_ROLE_LABEL_DETECTOR)

    def get_connected_calibrator_labels(self) -> list[str]:
        return self.get_connected_role_labels(role=COMPONENT_ROLE_LABEL_CALIBRATOR)

    def get_connected_pose_solver_labels(self) -> list[str]:
        return self.get_connected_role_labels(role=COMPONENT_ROLE_LABEL_POSE_SOLVER)

    def get_connected_role_labels(self, role: str) -> list[str]:
        return_value: list[str] = list()
        for connection in self._connections.values():
            if connection.static.role == role and connection.dynamic.status == "connected":
                return_value.append(connection.static.label)
        return return_value

    def ignore_request_and_response(
        self,
        client_identifier: str,
        request_id: uuid.UUID
    ):
        if client_identifier in self._request_series_by_label:
            for stored_request_index in range(len(self._request_series_by_label[client_identifier])):
                stored_request_id: uuid.UUID = \
                    self._request_series_by_label[client_identifier][stored_request_index][1]
                if stored_request_id == request_id:
                    self._request_series_by_label[client_identifier].pop(stored_request_index)
                    break
        if request_id in self._response_series_by_id:
            del self._response_series_by_id[request_id]

    def remove_connection(
        self,
        label: str
    ):
        if label not in self._connections:
            raise RuntimeError(f"Failed to find connection associated with {label}.")
        self._connections.pop(label)

    def request_series_push(
        self,
        connection_label: str,
        request_series: MCastRequestSeries
    ) -> uuid.UUID:
        if connection_label not in self._request_series_by_label:
            self._request_series_by_label[connection_label] = list()
        request_series_id: uuid.UUID = uuid.uuid4()
        self._request_series_by_label[connection_label].append((request_series, request_series_id))
        self._response_series_by_id[request_series_id] = None
        return request_series_id

    def response_series_pop(
        self,
        request_series_id: uuid.UUID
    ) -> MCastResponseSeries | None:
        """
        Only "pop" if there is a response (not None).
        Return value is the response series itself (or None)
        """
        if request_series_id not in self._response_series_by_id:
            raise ResponseSeriesNotExpected()

        if self._response_series_by_id[request_series_id] is None:
            return None

        response_series: MCastResponseSeries = self._response_series_by_id[request_series_id]
        self._response_series_by_id.pop(request_series_id)
        return response_series

    def supported_request_types(self) -> dict[type[MCastRequest], Callable[[dict], MCastResponse]]:
        return super().supported_request_types()

    async def do_update_frame_for_connection(
        self,
        connection: Connection
    ) -> None:
        if connection.dynamic.status == "disconnected" or connection.dynamic.status == "aborted":
            return

        if connection.dynamic.status == "disconnecting":
            if connection.dynamic.socket is not None:
                await connection.dynamic.socket.close()
                connection.dynamic.socket = None
            connection.dynamic.socket = None
            connection.dynamic.status = "disconnected"

        if connection.dynamic.status == "connecting":
            now_utc = datetime.datetime.utcnow()
            if now_utc >= connection.dynamic.next_attempt_timestamp_utc:
                connection.dynamic.attempt_count += 1
                uri: str = f"ws://{connection.static.ip_address}:{connection.static.port}/websocket"
                try:
                    connection.dynamic.socket = await connect(
                        uri=uri,
                        ping_timeout=None,
                        open_timeout=None,
                        close_timeout=None,
                        max_size=2**48)  # Default max_size might have trouble with some larger uncompressed images
                except ConnectionError as e:
                    if connection.dynamic.attempt_count >= BaseComponentConnectionDynamic.ATTEMPT_COUNT_MAXIMUM:
                        message = \
                            f"Failed to connect to {uri} with error: {str(e)}. "\
                            f"Connection is being aborted after {connection.dynamic.attempt_count} attempts."
                        self.add_status_message(severity="error", message=message)
                        connection.dynamic.status = "aborted"
                    else:
                        message: str = \
                            f"Failed to connect to {uri} with error: {str(e)}. "\
                            f"Will retry in {BaseComponentConnectionDynamic.ATTEMPT_TIME_GAP_SECONDS} seconds."
                        self.add_status_message(severity="warning", message=message)
                        connection.dynamic.next_attempt_timestamp_utc = now_utc + datetime.timedelta(
                            seconds=BaseComponentConnectionDynamic.ATTEMPT_TIME_GAP_SECONDS)
                    return
                message = f"Connected to {uri}."
                self.add_status_message(severity="info", message=message)
                connection.dynamic.status = "connected"
                connection.dynamic.attempt_count = 0

        if connection.dynamic.status == "connected":

            # TODO: Is this correct or even useful...?
            # if connection.dynamic.socket.closed:
            #     message = \
            #         f"Socket associated with {connection.static.label} appears to have been closed. "\
            #         f"Will attempt to reconnect."
            #     self.add_status_message(severity="warning", message=message)
            #     connection.dynamic.socket = None
            #     connection.dynamic.status = "connecting"
            #     connection.dynamic.attempt_count = 0
            #     return

            def response_series_converter(
                response_series_dict: dict
            ) -> MCastResponseSeries:
                series_list: list[MCastResponse] = self.parse_dynamic_series_list(
                    parsable_series_dict=response_series_dict,
                    supported_types=SUPPORTED_RESPONSE_TYPES)
                return MCastResponseSeries(series=series_list)

            # Handle manually-defined irregular tasks
            if connection.static.label in self._request_series_by_label:
                pairs: list[Tuple[MCastRequestSeries, uuid.UUID]] = \
                    self._request_series_by_label[connection.static.label]
                for pair in pairs:
                    response_series: MCastResponseSeries = \
                        await mcast_websocket_send_recv(
                            websocket=connection.dynamic.socket,
                            request_series=pair[0],
                            response_series_type=MCastResponseSeries,
                            response_series_converter=response_series_converter)
                    # TODO: This next line's logic may belong in the response_series_converter
                    response_series.responder = connection.static.label
                    self._response_series_by_id[pair[1]] = response_series
                self._request_series_by_label.pop(connection.static.label)

            # Regular every-frame stuff
            request_series: list[MCastRequest] = list()
            request_series.append(DequeueStatusMessagesRequest())

            response_series: MCastResponseSeries = await mcast_websocket_send_recv(
                websocket=connection.dynamic.socket,
                request_series=MCastRequestSeries(series=request_series),
                response_series_type=MCastResponseSeries,
                response_series_converter=response_series_converter)
            for response in response_series.series:
                if isinstance(response, DequeueStatusMessagesResponse):
                    for status_message in response.status_messages:
                        status_message_dict = status_message.dict()
                        status_message_dict["source_label"] = connection.static.label
                        self.add_status_message(**status_message_dict)

    async def do_update_frames_for_connections(
        self
    ) -> None:
        connections = list(self._connections.values())
        for connection in connections:
            await self.do_update_frame_for_connection(connection=connection)
