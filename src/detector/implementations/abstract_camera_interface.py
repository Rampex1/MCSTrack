import abc
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCastResponse
from src.detector.api import \
    GetCaptureDeviceResponse, \
    GetCapturePropertiesResponse, \
    GetCaptureImageResponse, \
    GetCaptureImageRequest
from src.common.structures.capture_status import CaptureStatus

import base64
import cv2
import datetime
import numpy

class AbstractCameraInterface(abc.ABC):

    _captured_timestamp_utc: datetime.datetime
    _capture_status: CaptureStatus  # internal bookkeeping

    def __del__(self):
        pass

    def internal_update_capture(self) -> None:
        pass

    def set_capture_device(self, **kwargs) -> EmptyResponse | ErrorResponse:
        pass
        
    def set_capture_properties(self, **kwargs) -> EmptyResponse:
        pass

    def get_capture_device(self, **_kwargs) -> GetCaptureDeviceResponse:
        pass

    def get_capture_properties(self, **_kwargs) -> GetCapturePropertiesResponse | ErrorResponse:
        pass

    def start_capture(self, **kwargs) -> MCastResponse:
        pass

    def stop_capture(self, **kwargs) -> MCastResponse:
        pass

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
