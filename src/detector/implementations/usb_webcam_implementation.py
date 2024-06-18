from src.detector.api import \
    GetCaptureDeviceResponse, \
    GetCapturePropertiesResponse, \
    SetCaptureDeviceRequest, \
    SetCapturePropertiesRequest
from src.detector.exceptions import UpdateCaptureError
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCastResponse
from src.common.structures.capture_status import CaptureStatus

from src.detector.implementations import AbstractCameraInterface

import cv2
import datetime
import logging
import numpy
import os

logger = logging.getLogger(__name__)

class USBWebcamWithOpenCV(AbstractCameraInterface):

    _capture: cv2.VideoCapture | None
    _captured_image: numpy.ndarray | None
    _captured_timestamp_utc: datetime.datetime
    _capture_device_id: str | int
    _capture_status: CaptureStatus  # internal bookkeeping

    def __init__(self, _capture_device_id):
        self._capture = None
        self._captured_image = None
        self._captured_timestamp_utc = datetime.datetime.min
        self._capture_device_id = _capture_device_id

        self._capture_status = CaptureStatus()
        self._capture_status.status = CaptureStatus.Status.STOPPED

    def __del__(self):
        if self._capture is not None:
            self._capture.release()

    def _detect_os_and_open_video(self,capture_device_id):
        # cv2.CAP_DSHOW does not work on linux, but is necessary on windows
        # The following is to check which platform we are on
        if os.name == 'nt':
            return cv2.VideoCapture(capture_device_id, cv2.CAP_DSHOW)
        else:
            return cv2.VideoCapture(capture_device_id)

    def internal_update_capture(self) -> tuple[str,str] | None:
        grabbed_frame: bool
        grabbed_frame = self._capture.grab()
        if not grabbed_frame:
            message: str = "Failed to grab frame."
            self._status.capture_errors.append(message)
            self._capture_status.status = CaptureStatus.Status.FAILURE
            raise UpdateCaptureError(severity="error", message=message)

        retrieved_frame: bool
        retrieved_frame, self._captured_image = self._capture.retrieve()
        if not retrieved_frame:
            message: str = "Failed to retrieve frame."
            logger.error(message)
            self._status.capture_errors.append(message)
            self._capture_status.status = CaptureStatus.Status.FAILURE
            raise UpdateCaptureError(severity="error", message=message)

        self._captured_timestamp_utc = datetime.datetime.utcnow()

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

    def start_capture(self, **kwargs) -> MCastResponse:
        if isinstance(self._capture_device_id, str) and self._capture_device_id.isnumeric():
            self._capture_device_id.isnumeric = int(self._capture_device_id)
        if self._capture is not None:
            return EmptyResponse()

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
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._capture_status.status = CaptureStatus.Status.STOPPED
        return EmptyResponse()
