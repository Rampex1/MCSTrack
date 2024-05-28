from src.detector.api import \
    GetCaptureDeviceRequest, \
    GetCaptureDeviceResponse, \
    GetCaptureImageRequest, \
    GetCaptureImageResponse, \
    GetCapturePropertiesRequest, \
    GetCapturePropertiesResponse, \
    GetDetectionParametersRequest, \
    GetDetectionParametersResponse, \
    GetMarkerSnapshotsRequest, \
    GetMarkerSnapshotsResponse, \
    SetCaptureDeviceRequest, \
    SetCapturePropertiesRequest, \
    SetDetectionParametersRequest, \
    StartCaptureRequest, \
    StopCaptureRequest
from src.detector.fileio import DetectorConfiguration
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCastComponent, \
    MCastRequest, \
    MCastResponse
from src.common.structures import \
    CornerRefinementMethod, \
    CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT, \
    CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT, \
    DetectionParameters, \
    MarkerSnapshot, \
    MarkerCornerImagePoint
from src.common.structures.capture_status import CaptureStatus
from src.common.structures.marker_status import MarkerStatus

import base64
import cv2.aruco
import datetime
import logging
import numpy
import os
from typing import Any, Callable

from src.detector.implementations import AbstractMarkerInterface

logger = logging.getLogger(__name__)

class Detector(MCastComponent):

    _detector_configuration: DetectorConfiguration

    #_camera_interface: AbstractCameraInterface
    _marker_interface: AbstractMarkerInterface

    _capture: cv2.VideoCapture | None
    _captured_image: numpy.ndarray | None
    _captured_timestamp_utc: datetime.datetime
    _capture_device_id: str | int

    _frame_count: int

    _capture_status: CaptureStatus  # internal bookkeeping

    def __init__(
        self,
        detector_configuration: DetectorConfiguration,
        marker_interface: AbstractMarkerInterface
    ):
        super().__init__(
            status_source_label=detector_configuration.serial_identifier,
            send_status_messages_to_logger=True)
        
        self._detector_configuration = detector_configuration


        # move to camera class
        self._capture = None
        self._captured_image = None
        self._captured_timestamp_utc = datetime.datetime.min
        self._capture_device_id = self._detector_configuration.camera_connection.usb_id

        self._frame_count = 0

        self._capture_status = CaptureStatus
        self._capture_status.status = CaptureStatus.Status.STOPPED
        
        #self._camera_interface = camera_interface
        self._marker_interface = marker_interface

    def __del__(self):
        if self._capture is not None:
            self._capture.release()

    def _detect_os_and_open_video(self,capture_device_id):
        if os.name == 'nt':
            return cv2.VideoCapture(capture_device_id, cv2.CAP_DSHOW)
        else:
            return cv2.VideoCapture(capture_device_id)

    async def internal_update(self):
        if self._capture_status.status == CaptureStatus.Status.RUNNING:
            self.internal_update_capture()
        if self._marker_interface.marker_status.status == MarkerStatus.Status.RUNNING and \
           self._captured_timestamp_utc > self._marker_interface.marker_timestamp_utc:
            self.internal_update_marker_corners()
        self._frame_count += 1
        if self._frame_count % 1000 == 0:
            print(f"Update count: {self._frame_count}")

    def internal_update_capture(self):
        grabbed_frame: bool
        grabbed_frame = self._capture.grab()
        if not grabbed_frame:
            message: str = "Failed to grab frame."
            self._status.capture_errors.append(message)
            self._capture_status.status = CaptureStatus.Status.FAILURE
            self.add_status_message(severity="error", message=message)
            return

        retrieved_frame: bool
        retrieved_frame, self._captured_image = self._capture.retrieve()
        if not retrieved_frame:
            message: str = "Failed to retrieve frame."
            logger.error(message)
            self._status.capture_errors.append(message)
            self._capture_status.status = CaptureStatus.Status.FAILURE
            return

        logger.debug("TEST")
        self._captured_timestamp_utc = datetime.datetime.utcnow()
    
    def supported_request_types(self) -> dict[type[MCastRequest], Callable[[dict], MCastResponse]]:
        return_value: dict[type[MCastRequest], Callable[[dict], MCastResponse]] = super().supported_request_types()
        return_value.update({
            GetCaptureDeviceRequest: self.get_capture_device,
            GetCaptureImageRequest: self.get_capture_image,
            GetCapturePropertiesRequest: self.get_capture_properties,
            GetDetectionParametersRequest: self.get_detection_parameters,
            GetMarkerSnapshotsRequest: self.get_marker_snapshots,
            SetCaptureDeviceRequest: self.set_capture_device,
            SetCapturePropertiesRequest: self.set_capture_properties,
            SetDetectionParametersRequest: self.set_detection_parameters,
            StartCaptureRequest: self.start_capture,
            StopCaptureRequest: self.stop_capture})
        return return_value
    
    def set_capture_device(self, **kwargs) -> EmptyResponse | ErrorResponse:
        """
        :key request: SetCaptureDeviceRequest
        """

        request: SetCaptureDeviceRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetCaptureDeviceRequest)

        input_device_id: int | str = request.capture_device_id
        if input_device_id.isnumeric():
            input_device_id = int(input_device_id)
        if self._capture_device_id != input_device_id:
            self._capture_device_id = input_device_id
            if self._capture is not None:
                self._capture.release()
                self._capture = self._detect_os_and_open_video(input_device_id)
                if not self._capture.isOpened():
                    return ErrorResponse(
                        message=f"Failed to open capture device {input_device_id}")
        return EmptyResponse()

    # noinspection DuplicatedCode
    def set_capture_properties(self, **kwargs) -> EmptyResponse:
        """
        :key request: SetCapturePropertiesRequest
        """

        request: SetCapturePropertiesRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetCapturePropertiesRequest)

        if self._capture is not None:
            if request.resolution_x_px is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(request.resolution_x_px))
            if request.resolution_y_px is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(request.resolution_y_px))
            if request.fps is not None:
                self._capture.set(cv2.CAP_PROP_FPS, float(request.fps))
            if request.auto_exposure is not None:
                self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(request.auto_exposure))
            if request.exposure is not None:
                self._capture.set(cv2.CAP_PROP_EXPOSURE, float(request.exposure))
            if request.brightness is not None:
                self._capture.set(cv2.CAP_PROP_BRIGHTNESS, float(request.brightness))
            if request.contrast is not None:
                self._capture.set(cv2.CAP_PROP_CONTRAST, float(request.contrast))
            if request.sharpness is not None:
                self._capture.set(cv2.CAP_PROP_SHARPNESS, float(request.sharpness))
            if request.gamma is not None:
                self._capture.set(cv2.CAP_PROP_GAMMA, float(request.gamma))
        return EmptyResponse()

    def get_capture_device(self, **_kwargs) -> GetCaptureDeviceResponse:
        return GetCaptureDeviceResponse(capture_device_id=str(self._capture_device_id))

    def get_capture_properties(self, **_kwargs) -> GetCapturePropertiesResponse | ErrorResponse:
        if self._capture is None:
            return ErrorResponse(
                message="The capture is not active, and properties cannot be retrieved.")
        else:
            return GetCapturePropertiesResponse(
                resolution_x_px=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                resolution_y_px=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=int(round(self._capture.get(cv2.CAP_PROP_FPS))),
                auto_exposure=bool(round(self._capture.get(cv2.CAP_PROP_AUTO_EXPOSURE))),
                exposure=int(self._capture.get(cv2.CAP_PROP_EXPOSURE)),
                brightness=int(self._capture.get(cv2.CAP_PROP_BRIGHTNESS)),
                contrast=int(self._capture.get(cv2.CAP_PROP_CONTRAST)),
                sharpness=int(self._capture.get(cv2.CAP_PROP_SHARPNESS)),
                gamma=int(self._capture.get(cv2.CAP_PROP_GAMMA)))
        # TODO: Get powerline_frequency_hz and backlight_compensation

    def get_capture_image(self, **kwargs) -> GetCaptureImageResponse:
        """
        :key request: GetCaptureImageRequest
        """

        request: GetCaptureImageRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=GetCaptureImageRequest)

        encoded_frame: bool
        encoded_image_rgb_single_row: numpy.array
        encoded, encoded_image_rgb_single_row = cv2.imencode(request.format, self._captured_image)
        encoded_image_rgb_bytes: bytes = encoded_image_rgb_single_row.tobytes()
        encoded_image_rgb_base64 = base64.b64encode(encoded_image_rgb_bytes)
        return GetCaptureImageResponse(
            format=request.format,
            image_base64=encoded_image_rgb_base64)

        # img_bytes = base64.b64decode(img_str)
        # img_buffer = numpy.frombuffer(img_bytes, dtype=numpy.uint8)
        # img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

    def start_capture(self, **kwargs) -> MCastResponse:
        """
        :key client_identifier: str
        """

        client_identifier: str = get_kwarg(
            kwargs=kwargs,
            key="client_identifier",
            arg_type=str)

        if isinstance(self._capture_device_id, str) and self._capture_device_id.isnumeric():
            self._capture_device_id.isnumeric = int(self._capture_device_id)
        if self._capture is not None:
            return EmptyResponse()
        self.add_status_message(severity="info", message=f"{client_identifier} is starting capture.")

        self._capture = self._detect_os_and_open_video(self._capture_device_id)
        # NOTE: The USB3 cameras bought for this project appear to require some basic parameters to be set,
        #       otherwise frame grab results in error
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture.set(cv2.CAP_PROP_FPS, 60)

        self._capture_status.status = CaptureStatus.Status.RUNNING
        return EmptyResponse()

    def stop_capture(self, **kwargs) -> MCastResponse:
        """
        :key client_identifier: str
        """

        client_identifier: str = get_kwarg(
            kwargs=kwargs,
            key="client_identifier",
            arg_type=str)

        self.add_status_message(severity="info", message=f"{client_identifier} is stopping capture.")
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._capture_status.status = CaptureStatus.Status.STOPPED
        return EmptyResponse()

    '''
    # Camera
    def set_capture_device(self, **kwargs) -> EmptyResponse | ErrorResponse:
        self._camera_interface.set_capture_device(**kwargs)
        
    def set_capture_properties(self, **kwargs) -> EmptyResponse:
        self._camera_interface.set_capture_properties(**kwargs)

    def get_capture_device(self, **_kwargs) -> GetCaptureDeviceResponse:
        # why underscored?
        self._camera_interface.get_capture_device(**_kwargs)

    def get_capture_properties(self, **_kwargs) -> GetCapturePropertiesResponse | ErrorResponse:
        self._camera_interface.get_capture_properties(**_kwargs)

    def get_capture_image(self, **kwargs) -> GetCaptureImageResponse:
        self._camera_interface.get_capture_image(**kwargs)

    def start_capture(self, **kwargs) -> MCastResponse:
        self._camera_interface.start_capture(**kwargs)

    def stop_capture(self, **kwargs) -> MCastResponse:
        self._camera_interface.stop_capture(**kwargs)'''
    
    # Marker
    def set_detection_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        return self._marker_interface.set_detection_parameters(**kwargs)

    def get_detection_parameters(self, **_kwargs) -> GetDetectionParametersResponse | ErrorResponse:
        return self._marker_interface.get_detection_parameters(**_kwargs)

    def get_marker_snapshots(self, **kwargs) -> GetMarkerSnapshotsResponse:
        return self._marker_interface.get_marker_snapshots(**kwargs)

    def internal_update_marker_corners(self):
        return self._marker_interface.internal_update_marker_corners(self._captured_image)

