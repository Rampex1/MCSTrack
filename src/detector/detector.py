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
from src.detector.exceptions import UpdateCaptureError
from src.detector.fileio import DetectorConfiguration
from src.calibrator import Calibrator
from src.calibrator.fileio import CalibratorConfiguration
from src.calibrator.api import \
    AddCalibrationImageRequest, \
    CalibrateRequest, \
    DeleteStagedRequest, \
    GetCalibrationImageRequest, \
    GetCalibrationResultRequest, \
    ListCalibrationDetectorResolutionsRequest, \
    ListCalibrationImageMetadataRequest, \
    ListCalibrationResultMetadataRequest, \
    UpdateCalibrationImageMetadataRequest, \
    UpdateCalibrationResultMetadataRequest
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    MCastComponent, \
    MCastRequest, \
    MCastResponse
from src.common.structures.capture_status import CaptureStatus
from src.common.structures.marker_status import MarkerStatus
import logging
from typing import Callable

from src.detector.implementations import \
    AbstractMarkerInterface, \
    AbstractCameraInterface

logger = logging.getLogger(__name__)

class Detector(MCastComponent):

    _detector_configuration: DetectorConfiguration
    _calibrator: Calibrator

    _camera_interface: AbstractCameraInterface
    _marker_interface: AbstractMarkerInterface

    _frame_count: int

    def __init__(
        self,
        detector_configuration: DetectorConfiguration,
        marker_interface: AbstractMarkerInterface,
        calibrator_configuration: CalibratorConfiguration,
        camera_interface: AbstractCameraInterface
    ):
        super().__init__(
            status_source_label=detector_configuration.serial_identifier,
            send_status_messages_to_logger=True)
        
        self._detector_configuration = detector_configuration
        self._calibrator = Calibrator(calibrator_configuration)
        self._frame_count = 0

        self._camera_interface = camera_interface
        self._marker_interface = marker_interface

    def __del__(self):
        self._camera_interface.__del__()

    async def internal_update(self):
        if self._camera_interface._capture_status.status == CaptureStatus.Status.RUNNING:
            self.internal_update_capture()
        if self._marker_interface.marker_status.status == MarkerStatus.Status.RUNNING and \
           self._camera_interface._captured_timestamp_utc > self._marker_interface.marker_timestamp_utc:
            self.internal_update_marker_corners()
        self._frame_count += 1
        if self._frame_count % 1000 == 0:
            print(f"Update count: {self._frame_count}")

    def supported_request_types(self) -> dict[type[MCastRequest], Callable[[dict], MCastResponse]]:
        return_value: dict[type[MCastRequest], Callable[[dict], MCastResponse]] = super().supported_request_types()
        return_value.update({

            # Detector Requests
            GetCaptureDeviceRequest: self.get_capture_device,
            GetCaptureImageRequest: self.get_capture_image,
            GetCapturePropertiesRequest: self.get_capture_properties,
            GetDetectionParametersRequest: self.get_detection_parameters,
            GetMarkerSnapshotsRequest: self.get_marker_snapshots,
            SetCaptureDeviceRequest: self.set_capture_device,
            SetCapturePropertiesRequest: self.set_capture_properties,
            SetDetectionParametersRequest: self.set_detection_parameters,
            StartCaptureRequest: self.start_capture,
            StopCaptureRequest: self.stop_capture,

            # Calibrator Requests
            AddCalibrationImageRequest: self._calibrator.add_calibration_image,
            CalibrateRequest: self._calibrator.calibrate,
            DeleteStagedRequest: self._calibrator.delete_staged,
            GetCalibrationImageRequest: self._calibrator.get_calibration_image,
            GetCalibrationResultRequest: self._calibrator.get_calibration_result,
            ListCalibrationDetectorResolutionsRequest: self._calibrator.list_calibration_detector_resolutions,
            ListCalibrationImageMetadataRequest: self._calibrator.list_calibration_image_metadata_list,
            ListCalibrationResultMetadataRequest: self._calibrator.list_calibration_result_metadata_list,
            UpdateCalibrationImageMetadataRequest: self._calibrator.update_calibration_image_metadata,
            UpdateCalibrationResultMetadataRequest: self._calibrator.update_calibration_result_metadata})
        return return_value
    
    # Camera
    def internal_update_capture(self):
        try:
            self._camera_interface.internal_update_capture()
        except UpdateCaptureError as e:
            self.add_status_message(
                severity=e.severity,
                message=e.message)

    def set_capture_device(self, **kwargs) -> EmptyResponse | ErrorResponse:
        return self._camera_interface.set_capture_device(**kwargs)
        
    def set_capture_properties(self, **kwargs) -> EmptyResponse:
        return self._camera_interface.set_capture_properties(**kwargs)

    def get_capture_device(self, **_kwargs) -> GetCaptureDeviceResponse:
        return self._camera_interface.get_capture_device(**_kwargs)

    def get_capture_properties(self, **_kwargs) -> GetCapturePropertiesResponse | ErrorResponse:
        return self._camera_interface.get_capture_properties(**_kwargs)

    def get_capture_image(self, **kwargs) -> GetCaptureImageResponse:
        return self._camera_interface.get_capture_image(**kwargs)

    def start_capture(self, **kwargs) -> MCastResponse:
        return self._camera_interface.start_capture(**kwargs)

    def stop_capture(self, **kwargs) -> MCastResponse:
        return self._camera_interface.stop_capture(**kwargs)
    
    # Marker
    def set_detection_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        return self._marker_interface.set_detection_parameters(**kwargs)

    def get_detection_parameters(self, **_kwargs) -> GetDetectionParametersResponse | ErrorResponse:
        return self._marker_interface.get_detection_parameters(**_kwargs)

    def get_marker_snapshots(self, **kwargs) -> GetMarkerSnapshotsResponse:
        return self._marker_interface.get_marker_snapshots(**kwargs)

    def internal_update_marker_corners(self):
        return self._marker_interface.internal_update_marker_corners(self._camera_interface._captured_image)
