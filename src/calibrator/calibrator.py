from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    ImageCoding, \
    MCastComponent, \
    MCastRequest, \
    MCastResponse
from src.common.structures import \
    DetectorResolution, \
    ImageResolution, \
    IntrinsicCalibration, \
    IntrinsicCalibrationFrameResult, \
    IntrinsicParameters, \
    Vec3
from src.common.util import \
    IOUtils
from src.calibrator.api import \
    AddCalibrationImageRequest, \
    AddCalibrationImageResponse, \
    CalibrateRequest, \
    CalibrateResponse, \
    DeleteStagedRequest, \
    GetCalibrationImageRequest, \
    GetCalibrationImageResponse, \
    GetCalibrationResultRequest, \
    GetCalibrationResultResponse, \
    ListCalibrationDetectorResolutionsRequest, \
    ListCalibrationDetectorResolutionsResponse, \
    ListCalibrationImageMetadataRequest, \
    ListCalibrationImageMetadataResponse, \
    ListCalibrationResultMetadataRequest, \
    ListCalibrationResultMetadataResponse, \
    UpdateCalibrationImageMetadataRequest, \
    UpdateCalibrationResultMetadataRequest
from src.calibrator.fileio import CalibratorConfiguration
from src.calibrator.structures import \
    CalibrationImageMetadata, \
    CalibrationImageState, \
    CalibrationMap, \
    CalibrationMapValue, \
    CalibrationResultMetadata, \
    CalibrationResultState, \
    CharucoBoardSpecification
import cv2
import cv2.aruco
import datetime
import json
from json import JSONDecodeError
import logging
import numpy
import os
from pydantic import ValidationError
from typing import Callable, Final
import uuid


logger = logging.getLogger(__name__)


class Calibrator:

    _configuration: CalibratorConfiguration
    _calibration_map: dict[DetectorResolution, CalibrationMapValue]

    CALIBRATION_MAP_FILENAME: Final[str] = "calibration_map.json"

    IMAGE_FORMAT: Final[str] = ".png"  # work in lossless image format
    RESULT_FORMAT: Final[str] = ".json"

    def __init__(
        self,
        calibrator_configuration: CalibratorConfiguration
    ):
        self._configuration = calibrator_configuration
        if not self._exists(path=self._configuration.data_path, pathtype="path", create_path=True):
            self.add_status_message(severity="critical", message="Data path does not exist and could not be created.")
            detailed_message: str = f"{self._configuration.data_path} does not exist and could not be created."
            logger.critical(detailed_message)
            raise RuntimeError(detailed_message)
        if not self._calibration_map_load():
            message: str = "The calibration map could not be loaded or created. "\
                           "In order to avoid data loss, the software will now abort. " \
                           "Please manually correct or remove the file in the filesystem."
            logger.critical(message)
            self.add_status_message(severity="critical", message=message)
            raise RuntimeError(message)

    def supported_request_types(self) -> dict[type[MCastRequest], Callable[[dict], MCastResponse]]:
        return_value: dict[type[MCastRequest], Callable[[dict], MCastResponse]] = super().supported_request_types()
        return return_value

    def add_calibration_image(self, **kwargs) -> AddCalibrationImageResponse | ErrorResponse:
        request: AddCalibrationImageRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=object)
        image_data: numpy.ndarray = ImageCoding.base64_to_image(input_base64=request.image_base64, color_mode="color")
        image_resolution: ImageResolution = ImageResolution(x_px=image_data.shape[1], y_px=image_data.shape[0])
        calibration_map_key: DetectorResolution = DetectorResolution(
            detector_serial_identifier=request.detector_serial_identifier,
            image_resolution=image_resolution)
        # Before making any changes to the calibration map, make sure folders exist,
        # and that this file does not somehow already exist (highly unlikely)
        key_path: str = self._path_for_map_key(calibration_map_key=calibration_map_key)
        if not self._exists(path=key_path, pathtype="path", create_path=True):
            return ErrorResponse(message=f"Failed to create storage location for input image.")
        image_identifier: str = str(uuid.uuid4())
        image_filepath = self._filepath_for_calibration_image(
            calibration_map_key=calibration_map_key,
            image_identifier=image_identifier)
        if os.path.exists(image_filepath):
            return ErrorResponse(
                message=f"Image {image_identifier} appears to already exist. This is never expected to occur. "
                        f"Please try again, and if this error continues to occur then please report a bug.")
        if calibration_map_key not in self._calibration_map:
            self._calibration_map[calibration_map_key] = CalibrationMapValue()
        self._calibration_map[calibration_map_key].image_metadata_list.append(
            CalibrationImageMetadata(identifier=image_identifier))
        # noinspection PyTypeChecker
        image_bytes = ImageCoding.image_to_bytes(image_data=image_data, image_format=Calibrator.IMAGE_FORMAT)
        with (open(image_filepath, 'wb') as in_file):
            in_file.write(image_bytes)
        self._calibration_map_save()
        return AddCalibrationImageResponse(image_identifier=image_identifier)

    def calibrate(self, **kwargs) -> CalibrateResponse | ErrorResponse:
        request: CalibrateRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=CalibrateRequest)

        calibration_key: DetectorResolution = DetectorResolution(
            detector_serial_identifier=request.detector_serial_identifier,
            image_resolution=request.image_resolution)
        if calibration_key not in self._calibration_map:
            return ErrorResponse(
                message=f"No images for given detector {request.detector_serial_identifier} "
                        f"and resolution {str(request.image_resolution)} found.")

        # TODO: Instead of detecting the markers here, maybe the detector could do it instead (just send the points/ids)

        # TODO: Detection parameters to come from the detector, somehow
        aruco_detector_parameters = cv2.aruco.DetectorParameters_create()

        # TODO: ChArUco board to come from somewhere (user???)
        charuco_spec = CharucoBoardSpecification()
        # noinspection PyUnresolvedReferences
        charuco_board: cv2.aruco.CharucoBoard = charuco_spec.create_board()

        calibration_value: CalibrationMapValue = self._calibration_map[calibration_key]
        all_charuco_corners = list()
        all_charuco_ids = list()
        image_identifiers: list[str] = list()
        for image_metadata in calibration_value.image_metadata_list:
            if image_metadata.state != CalibrationImageState.SELECT:
                continue
            image_filepath: str = self._filepath_for_calibration_image(
                calibration_map_key=calibration_key,
                image_identifier=image_metadata.identifier)
            if not self._exists(path=image_filepath, pathtype="filepath"):
                self.add_status_message(
                    severity="warning",
                    message=f"Image {image_metadata.identifier} was not found. "
                            f"It will be omitted from the calibration.")
                continue
            image_rgb = cv2.imread(image_filepath)
            image_greyscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            (marker_corners, marker_ids, _) = cv2.aruco.detectMarkers(
                image=image_greyscale,
                dictionary=charuco_spec.aruco_dictionary(),
                parameters=aruco_detector_parameters)
            if len(marker_corners) <= 0:
                self.add_status_message(
                    severity="warning",
                    message=f"Image {image_metadata.identifier} did not appear to contain any identifiable markers. "
                            f"It will be omitted from the calibration.")
                continue
            image_identifiers.append(image_metadata.identifier)
            # Note:
            # Marker corners are the corners of the markers, whereas
            # ChArUco corners are the corners of the chessboard.
            # ChArUco calibration function works with the corners of the chessboard.
            _, frame_charuco_corners, frame_charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=image_greyscale,
                board=charuco_board,
            )
            # Algorithm requires a minimum of 4 markers
            if len(frame_charuco_corners) >= 4:
                all_charuco_corners.append(frame_charuco_corners)
                all_charuco_ids.append(frame_charuco_ids)

        if len(all_charuco_corners) <= 0:
            return ErrorResponse(message="The input images did not contain visible markers.")

        # outputs to be stored in these containers
        calibration_result = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=charuco_board,
            imageSize=numpy.array(charuco_spec.size_mm(), dtype="int32"),  # Exception if float
            cameraMatrix=numpy.identity(3, dtype='f'),
            distCoeffs=numpy.zeros(5, dtype='f'))

        charuco_overall_reprojection_error = calibration_result[0]
        charuco_camera_matrix = calibration_result[1]
        charuco_distortion_coefficients = calibration_result[2]
        charuco_rotation_vectors = calibration_result[3]
        charuco_translation_vectors = calibration_result[4]
        charuco_intrinsic_stdevs = calibration_result[5]
        charuco_extrinsic_stdevs = calibration_result[6]
        charuco_reprojection_errors = calibration_result[7]

        # TODO: Assertion on size of distortion coefficients being 5?
        #  Note: OpenCV documentation specifies the order of distortion coefficients
        #  https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#ga366993d29fdddd995fba8c2e6ca811ea
        #  So far I have not seen calibration return a number of coefficients other than 5.

        intrinsic_calibration: IntrinsicCalibration = IntrinsicCalibration(
            timestamp_utc=str(datetime.datetime.utcnow()),
            detector_serial_identifier=request.detector_serial_identifier,
            image_resolution=request.image_resolution,
            calibrated_values=IntrinsicParameters(
                focal_length_x_px=charuco_camera_matrix[0, 0],
                focal_length_y_px=charuco_camera_matrix[1, 1],
                optical_center_x_px=charuco_camera_matrix[0, 2],
                optical_center_y_px=charuco_camera_matrix[1, 2],
                radial_distortion_coefficients=[
                    charuco_distortion_coefficients[0, 0],
                    charuco_distortion_coefficients[1, 0],
                    charuco_distortion_coefficients[4, 0]],
                tangential_distortion_coefficients=[
                    charuco_distortion_coefficients[2, 0],
                    charuco_distortion_coefficients[3, 0]]),
            calibrated_stdevs=[value[0] for value in charuco_intrinsic_stdevs],
            reprojection_error=charuco_overall_reprojection_error,
            frame_results=[
                IntrinsicCalibrationFrameResult(
                    image_identifier=image_identifiers[i],
                    translation=Vec3(
                        x=charuco_translation_vectors[i][0, 0],
                        y=charuco_translation_vectors[i][1, 0],
                        z=charuco_translation_vectors[i][2, 0]),
                    rotation=Vec3(
                        x=charuco_rotation_vectors[i][0, 0],
                        y=charuco_rotation_vectors[i][1, 0],
                        z=charuco_rotation_vectors[i][2, 0]),
                    translation_stdev=Vec3(
                        x=charuco_extrinsic_stdevs[i*6 + 3, 0],
                        y=charuco_extrinsic_stdevs[i*6 + 4, 0],
                        z=charuco_extrinsic_stdevs[i*6 + 5, 0]),
                    rotation_stdev=Vec3(
                        x=charuco_extrinsic_stdevs[i*6 + 0, 0],
                        y=charuco_extrinsic_stdevs[i*6 + 1, 0],
                        z=charuco_extrinsic_stdevs[i*6 + 2, 0]),
                    reprojection_error=charuco_reprojection_errors[i, 0])
                for i in range(0, len(charuco_reprojection_errors))])

        result_identifier: str = str(uuid.uuid4())
        result_filepath = self._filepath_for_calibration_result(
            calibration_map_key=calibration_key,
            result_identifier=result_identifier)
        IOUtils.json_write(
            filepath=result_filepath,
            json_dict=intrinsic_calibration.dict(),
            on_error_for_user=lambda msg: self.add_status_message(severity="error", message=msg),
            on_error_for_dev=logger.error,
            ignore_none=True)
        result_metadata: CalibrationResultMetadata = CalibrationResultMetadata(
            identifier=result_identifier,
            image_identifiers=image_identifiers)
        self._calibration_map[calibration_key].result_metadata_list.append(result_metadata)
        self._calibration_map_save()
        return CalibrateResponse(
            result_identifier=result_identifier,
            intrinsic_calibration=intrinsic_calibration)

    def delete_staged(self, **_kwargs) -> EmptyResponse:
        for calibration_key in self._calibration_map.keys():
            calibration_value: CalibrationMapValue = self._calibration_map[calibration_key]
            image_indices_to_delete: list = list()
            for image_index, image in enumerate(calibration_value.image_metadata_list):
                if image.state == CalibrationImageState.DELETE:
                    self._filepath_delete_if_exists(self._filepath_for_calibration_image(
                        calibration_map_key=calibration_key,
                        image_identifier=image.identifier))
                    image_indices_to_delete.append(image_index)
            for i in reversed(image_indices_to_delete):
                del calibration_value.image_metadata_list[i]
            result_indices_to_delete: list = list()
            for result_index, result in enumerate(calibration_value.result_metadata_list):
                if result.state == CalibrationResultState.DELETE:
                    self._filepath_delete_if_exists(self._filepath_for_calibration_result(
                        calibration_map_key=calibration_key,
                        result_identifier=result.identifier))
                    result_indices_to_delete.append(result_index)
            for i in reversed(result_indices_to_delete):
                del calibration_value.result_metadata_list[i]
        self._calibration_map_save()
        return EmptyResponse()

    # noinspection DuplicatedCode
    def get_calibration_image(self, **kwargs) -> GetCalibrationImageResponse | ErrorResponse:
        request: GetCalibrationImageRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=GetCalibrationImageRequest)
        found_count: int = 0
        matching_detector_resolution: DetectorResolution | None = None
        for detector_resolution in self._calibration_map:
            for image in self._calibration_map[detector_resolution].image_metadata_list:
                if image.identifier == request.image_identifier:
                    found_count += 1
                    matching_detector_resolution = detector_resolution
                    break
        if found_count < 1:
            return ErrorResponse(
                message=f"Image identifier {request.image_identifier} is not associated with any image.")
        elif found_count > 1:
            return ErrorResponse(
                message=f"Image identifier {request.image_identifier} is associated with multiple images.")

        image_filepath = self._filepath_for_calibration_image(
            calibration_map_key=matching_detector_resolution,
            image_identifier=request.image_identifier)
        if not os.path.exists(image_filepath):
            return ErrorResponse(
                message=f"File does not exist for image {request.image_identifier} "
                        f"and given detector {matching_detector_resolution.detector_serial_identifier} "
                        f"and resolution {str(matching_detector_resolution.image_resolution)}.")
        image_bytes: bytes
        try:
            with (open(image_filepath, 'rb') as in_file):
                image_bytes = in_file.read()
        except OSError:
            return ErrorResponse(
                message=f"Failed to open image {request.image_identifier} for "
                        f"given detector {matching_detector_resolution.detector_serial_identifier} "
                        f"and resolution {str(matching_detector_resolution.image_resolution)}.")
        image_base64 = ImageCoding.bytes_to_base64(image_bytes=image_bytes)
        return GetCalibrationImageResponse(image_base64=image_base64)

    # noinspection DuplicatedCode
    def get_calibration_result(self, **kwargs) -> GetCalibrationResultResponse | ErrorResponse:
        request: GetCalibrationResultRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=GetCalibrationResultRequest)
        found_count: int = 0
        matching_detector_resolution: DetectorResolution | None = None
        for detector_resolution in self._calibration_map:
            for result in self._calibration_map[detector_resolution].result_metadata_list:
                if result.identifier == request.result_identifier:
                    found_count += 1
                    matching_detector_resolution = detector_resolution
                    break
        if found_count < 1:
            return ErrorResponse(
                message=f"Image identifier {request.result_identifier} is not associated with any image.")
        elif found_count > 1:
            return ErrorResponse(
                message=f"Image identifier {request.result_identifier} is associated with multiple images.")

        result_filepath = self._filepath_for_calibration_result(
            calibration_map_key=matching_detector_resolution,
            result_identifier=request.result_identifier)
        if not os.path.exists(result_filepath):
            return ErrorResponse(
                message=f"File does not exist for result {request.result_identifier} "
                        f"and given detector {matching_detector_resolution.detector_serial_identifier} "
                        f"and resolution {str(matching_detector_resolution.image_resolution)}.")
        result_json_raw: str
        try:
            with (open(result_filepath, 'r') as in_file):
                result_json_raw = in_file.read()
        except OSError:
            return ErrorResponse(
                message=f"Failed to open result {request.result_identifier} for "
                        f"given detector {matching_detector_resolution.detector_serial_identifier} "
                        f"and resolution {str(matching_detector_resolution.image_resolution)}.")
        result_json_dict: dict
        try:
            result_json_dict = dict(json.loads(result_json_raw))
        except JSONDecodeError:
            return ErrorResponse(
                message=f"Failed to parse result {request.result_identifier} for "
                        f"given detector {matching_detector_resolution.detector_serial_identifier} "
                        f"and resolution {str(matching_detector_resolution.image_resolution)}.")
        intrinsic_calibration: IntrinsicCalibration = IntrinsicCalibration(**result_json_dict)
        return GetCalibrationResultResponse(intrinsic_calibration=intrinsic_calibration)

    def list_calibration_detector_resolutions(self, **_kwargs) -> ListCalibrationDetectorResolutionsResponse:
        detector_resolutions: list[DetectorResolution] = list(self._calibration_map.keys())
        return ListCalibrationDetectorResolutionsResponse(detector_resolutions=detector_resolutions)

    # noinspection DuplicatedCode
    def list_calibration_image_metadata_list(self, **kwargs) -> ListCalibrationImageMetadataResponse:
        request: ListCalibrationImageMetadataRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=ListCalibrationImageMetadataRequest)
        calibration_map_key: DetectorResolution = DetectorResolution(
            detector_serial_identifier=request.detector_serial_identifier,
            image_resolution=request.image_resolution)
        image_metadata_list: list[CalibrationImageMetadata] = list()
        if calibration_map_key in self._calibration_map:
            image_metadata_list = [image
                                   for image in self._calibration_map[calibration_map_key].image_metadata_list]
        return ListCalibrationImageMetadataResponse(metadata_list=image_metadata_list)

    # noinspection DuplicatedCode
    def list_calibration_result_metadata_list(self, **kwargs) -> ListCalibrationResultMetadataResponse:
        request: ListCalibrationResultMetadataRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=ListCalibrationResultMetadataRequest)
        calibration_map_key: DetectorResolution = DetectorResolution(
            detector_serial_identifier=request.detector_serial_identifier,
            image_resolution=request.image_resolution)
        result_metadata_list: list[CalibrationResultMetadata] = list()
        if calibration_map_key in self._calibration_map:
            result_metadata_list = [result
                                    for result in self._calibration_map[calibration_map_key].result_metadata_list]
        return ListCalibrationResultMetadataResponse(metadata_list=result_metadata_list)

    # noinspection DuplicatedCode
    def update_calibration_image_metadata(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: UpdateCalibrationImageMetadataRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=UpdateCalibrationImageMetadataRequest)
        found_count: int = 0
        for calibration_map_key in self._calibration_map:
            for image in self._calibration_map[calibration_map_key].image_metadata_list:
                if image.identifier == request.image_identifier:
                    image.state = request.image_state
                    image.label = request.image_label
                    found_count += 1
                    break
        if found_count < 1:
            return ErrorResponse(
                message=f"Image identifier {request.image_identifier} is not associated with any image.")
        elif found_count > 1:
            self.add_status_message(
                severity="warning",
                message=f"Image identifier {request.image_identifier} is associated with multiple images.")
        self._calibration_map_save()
        return EmptyResponse()

    # noinspection DuplicatedCode
    def update_calibration_result_metadata(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: UpdateCalibrationResultMetadataRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=UpdateCalibrationResultMetadataRequest)
        found_count: int = 0
        for calibration_map_key in self._calibration_map:
            for result in self._calibration_map[calibration_map_key].result_metadata_list:
                if result.identifier == request.result_identifier:
                    result.state = request.result_state
                    found_count += 1
                    break
        if found_count < 1:
            return ErrorResponse(
                message=f"Result identifier {request.result_identifier} is not associated with any result.")
        elif found_count > 1:
            self.add_status_message(
                severity="warning",
                message=f"Result identifier {request.result_identifier} is associated with multiple results.")
        self._calibration_map_save()
        return EmptyResponse()

    def _filepath_for_calibration_image(
        self,
        calibration_map_key: DetectorResolution,
        image_identifier: str
    ) -> str:
        key_path: str = self._path_for_map_key(calibration_map_key=calibration_map_key)
        return os.path.join(
            key_path,
            image_identifier + Calibrator.IMAGE_FORMAT)

    def _filepath_for_calibration_result(
        self,
        calibration_map_key: DetectorResolution,
        result_identifier: str
    ) -> str:
        key_path: str = self._path_for_map_key(calibration_map_key=calibration_map_key)
        return os.path.join(
            key_path,
            result_identifier + Calibrator.RESULT_FORMAT)

    def _filepath_delete_if_exists(self, filepath: str):
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            logger.error(e)
            self.add_status_message(
                severity="warning",  # It *is* an internal error, but it has few consequences for user... so warning
                message=f"Failed to remove a file from the calibrator because it does not exist. "
                        f"See its internal log for details.")
        except OSError as e:
            logger.error(e)
            self.add_status_message(
                severity="warning",  # It *is* an internal error, but it has few consequences for user... so warning
                message=f"Failed to remove a file from the calibrator due to an unexpected reason. "
                        f"See its internal log for details.")

    def _path_for_map_key(
        self,
        calibration_map_key: DetectorResolution
    ) -> str:
        return os.path.join(
            self._configuration.data_path,
            str(DetectorResolution(
                detector_serial_identifier=calibration_map_key.detector_serial_identifier,
                image_resolution=calibration_map_key.image_resolution)))

    def _exists(
        self,
        path: str,
        pathtype: IOUtils.PathType,
        create_path: bool = False
    ) -> bool:
        return IOUtils.exists(
            path=path,
            pathtype=pathtype,
            create_path=create_path,
            on_error_for_user=lambda msg: self.add_status_message(severity="error", message=msg),
            on_error_for_dev=logger.error)

    def _calibration_map_filepath(self) -> str:
        return os.path.join(self._configuration.data_path, Calibrator.CALIBRATION_MAP_FILENAME)

    def _calibration_map_load(self) -> bool:
        """
        :return: True if loaded or if it can be created without overwriting existing data. False otherwise.
        """
        calibration_map_filepath: str = self._calibration_map_filepath()
        if not os.path.exists(calibration_map_filepath):
            self._calibration_map = dict()
            return True
        elif not os.path.isfile(calibration_map_filepath):
            logger.critical(f"Calibration map file location {calibration_map_filepath} exists but is not a file.")
            self.add_status_message(
                severity="critical",
                message="Filepath location for calibration map exists but is not a file. "
                        "Most likely a directory exists at that location, "
                        "and it needs to be manually removed.")
            return False
        json_dict: dict = IOUtils.hjson_read(
            filepath=calibration_map_filepath,
            on_error_for_user=lambda msg: self.add_status_message(severity="error", message=msg),
            on_error_for_dev=logger.error)
        if not json_dict:
            logger.error(f"Failed to load calibration map from file {calibration_map_filepath}.")
            self.add_status_message(
                severity="error",
                message="Failed to load calibration map from file.")
            return False
        calibration_map: CalibrationMap
        try:
            calibration_map = CalibrationMap(**json_dict)
        except ValidationError as e:
            logger.error(e)
            self.add_status_message(
                severity="error",
                message="Failed to parse calibration map from file.")
            return False
        self._calibration_map = calibration_map.as_dict()
        return True

    def _calibration_map_save(self) -> None:
        IOUtils.json_write(
            filepath=self._calibration_map_filepath(),
            json_dict=CalibrationMap.from_dict(self._calibration_map).dict(),
            on_error_for_user=lambda msg: self.add_status_message(severity="error", message=msg),
            on_error_for_dev=logger.error)
