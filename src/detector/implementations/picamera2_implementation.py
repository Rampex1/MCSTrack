from src.detector.api import \
    GetCaptureDeviceResponse, \
    GetCapturePropertiesResponse, \
    SetCapturePropertiesRequest
from src.detector.exceptions import UpdateCaptureError
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCastResponse
from src.common.structures.capture_status import CaptureStatus

from src.detector.implementations import AbstractCameraInterface

from picamera2 import Picamera2
from picamera2.controls import Controls

import datetime
import logging
import numpy

logger = logging.getLogger(__name__)

class PiCamera(AbstractCameraInterface):

    _camera: Picamera2
    _captured_image: numpy.ndarray | None
    _captured_timestamp_utc: datetime.datetime
    _capture_status: CaptureStatus  # internal bookkeeping
    _camera_controls: Controls

    def __init__(self):
        self._captured_image = None
        self._captured_timestamp_utc = datetime.datetime.min

        self._capture_status = CaptureStatus()
        self._capture_status.status = CaptureStatus.Status.STOPPED

        self._camera = Picamera2()
        self._camera.configure("video")

        default_value_index: int = 2
        default_brightness = self._camera.camera_controls['Brightness'][default_value_index]
        default_contrast = self._camera.camera_controls['Contrast'][default_value_index]
        default_sharpness = self._camera.camera_controls['Sharpness'][default_value_index]
        default_auto_exposure = self._camera.camera_controls['AeEnable'][default_value_index]
        default_exposure = self._camera.camera_controls['ExposureValue'][default_value_index]

        self._camera_controls = Controls(self._camera)

        self._camera_controls.Brightness = default_brightness
        self._camera_controls.Contrast = default_contrast
        self._camera_controls.Sharpness = default_sharpness
        self._camera_controls.AeEnable = default_auto_exposure
        self._camera_controls.ExposureValue = default_exposure

        self._camera.set_controls(self._camera_controls)

    def __del__(self):
        if self._captured_image is not None:
            self._captured_image = numpy.empty()

    def internal_update_capture(self) -> None:
        self._captured_image = self._camera.capture_array()

        if self._captured_image is None:
            message: str = "Failed to grab frame."
            self._status.capture_errors.append(message)
            self._capture_status.status = CaptureStatus.Status.FAILURE
            raise UpdateCaptureError(severity="error", message=message)

        self._captured_timestamp_utc = datetime.datetime.utcnow()

    def set_capture_device(self, **kwargs) -> EmptyResponse | ErrorResponse:
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

        if self._captured_image is not None:

            self._camera.stop()
            if request.resolution_x_px is not None and request.resolution_y_px is not None:
                self._camera.video_configuration.size = (request.resolution_x_px,request.resolution_y_px)
            if request.fps is not None:
                self._camera.video_configuration.controls.FrameRate = request.fps
            self._camera.configure("video")

            if request.auto_exposure is not None:
                self._camera_controls.AeEnable = request.auto_exposure
            # TODO: how to enforce values in gui be entered in the proper range?
            if request.exposure is not None:
                self._camera_controls.ExposureValue = request.exposure
            if request.brightness is not None:
                self._camera_controls.Brightness = request.brightness
            if request.contrast is not None:
                self._camera_controls.Contrast = request.contrast
            if request.sharpness is not None:
                self._camera_controls.Sharpness = request.sharpness

            self._camera.set_controls(self._camera_controls)
            self._camera.start()
        return EmptyResponse()

    def get_capture_device(self, **_kwargs) -> GetCaptureDeviceResponse:
        return GetCaptureDeviceResponse(capture_device_id=str("N/A"))

    def get_capture_properties(self, **_kwargs) -> GetCapturePropertiesResponse | ErrorResponse:
        if self._captured_image is None:
            return ErrorResponse(
                message="The capture is not active, and properties cannot be retrieved.")
        else:
            ret = GetCapturePropertiesResponse(
                resolution_x_px=int(self._camera.video_configuration.size[0]),
                resolution_y_px=int(self._camera.video_configuration.size[1]),
                fps=int(round(self._camera.video_configuration.controls.FrameRate)),
                brightness=self._camera_controls.Brightness,
                auto_exposure=bool(self._camera_controls.AeEnable),
                exposure=int(self._camera_controls.ExposureValue),
                contrast=self._camera_controls.Contrast,
                sharpness=self._camera_controls.Sharpness)
            return ret

    def start_capture(self, **kwargs) -> MCastResponse:
        self._camera.start()
        self._captured_image = self._camera.capture_array()
        self._capture_status.status = CaptureStatus.Status.RUNNING
        return EmptyResponse()

    def stop_capture(self, **kwargs) -> MCastResponse:
        if self._captured_image is not None:
            self._captured_image = None
        self._capture_status.status = CaptureStatus.Status.STOPPED
        self._camera.stop()
        return EmptyResponse()
